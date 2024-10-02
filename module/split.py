import os
import argparse
import math
import shutil  # ファイルをコピーするために使用

class VideoFrameExtractor:
    def __init__(self, frames_folder, output_base_dir='output_frames', duration=60, interval=10):
        self.frames_folder = frames_folder
        self.output_base_dir = output_base_dir
        self.duration = duration  # セグメントの長さ（秒）
        self.interval = interval  # インターバル（秒）

        # 出力フォルダの作成
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)

        # フレームファイルのリストを取得
        self.frame_files = sorted(os.listdir(self.frames_folder))

        # ファイル名からタイムスタンプを抽出
        self.timestamps = []
        self.frame_dict = {}
        for filename in self.frame_files:
            name, ext = os.path.splitext(filename)
            try:
                timestamp = int(name)
                self.timestamps.append(timestamp)
                self.frame_dict[timestamp] = filename
            except ValueError:
                pass  # 数値でないファイル名はスキップ

        # タイムスタンプをソート
        self.timestamps.sort()

        # フレームが存在するか確認
        if not self.timestamps:
            raise ValueError("指定されたフォルダに有効なフレームファイルがありません。")

        # ベースタイムスタンプを取得
        self.base_timestamp = self.timestamps[0]

        # 動画の総長さを計算
        self.video_duration = (self.timestamps[-1] - self.timestamps[0]) / 1000.0  # 秒単位

        # セグメント数を計算
        self.num_segments = math.ceil((self.video_duration - self.duration) / self.interval) + 1
        self.num_segments = max(self.num_segments, 1)

    def extract_frames_segment_with_base(self, start_time, duration, segment_output_dir, segment_base_timestamp):
        if not os.path.exists(segment_output_dir):
            os.makedirs(segment_output_dir)

        # セグメントの開始・終了タイムスタンプをミリ秒で計算
        start_timestamp = segment_base_timestamp
        end_timestamp = start_timestamp + int(duration * 1000)

        # 該当するフレームをセグメントフォルダにコピー
        for timestamp in self.timestamps:
            if start_timestamp <= timestamp <= end_timestamp:
                filename = self.frame_dict[timestamp]
                src_path = os.path.join(self.frames_folder, filename)
                dst_path = os.path.join(segment_output_dir, filename)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    print(f"警告: フレーム {filename} が存在しません。")
            elif timestamp > end_timestamp:
                break  # これ以上のチェックは不要

    def extract_frames(self):
        for i in range(self.num_segments):
            start_time = i * self.interval
            end_time = start_time + self.duration

            # 動画の長さを超えないように終了時間を調整
            if end_time > self.video_duration:
                end_time = self.video_duration

            actual_duration = end_time - start_time

            # 開始時間が動画の長さを超える場合は処理を終了
            if start_time >= self.video_duration:
                break

            segment_output_dir = os.path.join(self.output_base_dir, f'segment_{i}')

            # セグメントごとのベースタイムスタンプ
            segment_base_timestamp = self.base_timestamp + int(start_time * 1000)

            # セグメントのフレームを抽出
            self.extract_frames_segment_with_base(start_time, actual_duration, segment_output_dir, segment_base_timestamp)

        # 最後のフレームが含まれているか確認し、含まれていなければ最後のセグメントに追加
        last_frame_timestamp = self.timestamps[-1]
        last_frame_included = False
        for i in range(self.num_segments):
            segment_output_dir = os.path.join(self.output_base_dir, f'segment_{i}')
            if os.path.exists(os.path.join(segment_output_dir, self.frame_dict[last_frame_timestamp])):
                last_frame_included = True
                break

        if not last_frame_included:
            # 最後のフレームを最後のセグメントにコピー
            segment_output_dir = os.path.join(self.output_base_dir, f'segment_{self.num_segments -1}')
            if not os.path.exists(segment_output_dir):
                os.makedirs(segment_output_dir)
            filename = self.frame_dict[last_frame_timestamp]
            src_path = os.path.join(self.frames_folder, filename)
            dst_path = os.path.join(segment_output_dir, filename)
            shutil.copy(src_path, dst_path)

# メイン部分
if __name__ == '__main__':
    # コマンドライン引数をパース
    parser = argparse.ArgumentParser(description='Video Frame Extractor')
    parser.add_argument('--frames_folder', type=str, required=True, help='入力フレームが格納されたフォルダを指定')
    parser.add_argument('--output_base_dir', type=str, default='output_frames', help='出力フォルダのベースパス')
    parser.add_argument('--duration', type=int, default=60, help='セグメントの長さ（秒）')
    parser.add_argument('--interval', type=int, default=10, help='インターバル（秒）')
    args = parser.parse_args()

    extractor = VideoFrameExtractor(
        frames_folder=args.frames_folder,
        output_base_dir=args.output_base_dir,
        duration=args.duration,
        interval=args.interval
    )
    extractor.extract_frames()
