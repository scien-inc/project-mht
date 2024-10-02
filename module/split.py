import os
import cv2
import argparse  # argparseをインポート

class VideoFrameExtractor:
    def __init__(self, video_path, output_base_dir='output_frames', duration=60, interval=10):
        self.video_path = video_path
        self.output_base_dir = output_base_dir
        self.duration = duration  # セグメントの長さ（秒）
        self.interval = interval  # インターバル（秒）

        # 入力動画ファイル名からベースタイムスタンプを取得（ミリ秒）
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.base_timestamp = int(base_name)

        # 出力フォルダの作成
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)

        # 動画のプロパティを取得
        self.vidcap = cv2.VideoCapture(self.video_path)
        self.original_fps = self.vidcap.get(cv2.CAP_PROP_FPS)
        self.total_frames = self.vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_duration = self.total_frames / self.original_fps * 1000  # ミリ秒単位に変換
        self.vidcap.release()

        # セグメント数を計算
        self.num_segments = int((self.video_duration / 1000 - self.duration) / self.interval) + 1
        self.num_segments = max(self.num_segments, 1)

    def extract_frames_segment(self, start_time, duration, segment_output_dir):
        if not os.path.exists(segment_output_dir):
            os.makedirs(segment_output_dir)

        vidcap = cv2.VideoCapture(self.video_path)

        current_timestamp_in_video = start_time * 1000  # スタート時間をミリ秒に変換
        end_timestamp_in_video = (start_time + duration) * 1000  # 終了時間をミリ秒に変換

        while current_timestamp_in_video <= end_timestamp_in_video:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, current_timestamp_in_video)
            success, image = vidcap.read()
            if success:
                # 出力ファイル名を計算
                output_timestamp = self.base_timestamp + int(current_timestamp_in_video - start_time * 1000)
                filename = f"{output_timestamp}.jpg"
                cv2.imwrite(os.path.join(segment_output_dir, filename), image)
            else:
                break
            current_timestamp_in_video += 200  # 200ミリ秒ずつ増加

        vidcap.release()

    def extract_frames(self):
        for i in range(self.num_segments):
            start_time = i * self.interval
            end_time = start_time + self.duration
            actual_duration = self.duration

            if end_time > self.video_duration / 1000:
                end_time = self.video_duration / 1000
                actual_duration = end_time - start_time

            segment_output_dir = os.path.join(self.output_base_dir, f'segment_{i}')
            self.extract_frames_segment(start_time, actual_duration, segment_output_dir)

# メイン部分
if __name__ == '__main__':
    # コマンドライン引数をパース
    parser = argparse.ArgumentParser(description='Video Frame Extractor')
    parser.add_argument('--video_path', type=str, required=True, help='入力動画のパスを指定（ファイル名がミリ秒）')
    parser.add_argument('--output_base_dir', type=str, default='output_frames', help='出力フォルダのベースパス')
    parser.add_argument('--duration', type=int, default=60, help='セグメントの長さ（秒）')
    parser.add_argument('--interval', type=int, default=10, help='インターバル（秒）')
    args = parser.parse_args()

    extractor = VideoFrameExtractor(
        video_path=args.video_path,
        output_base_dir=args.output_base_dir,
        duration=args.duration,
        interval=args.interval
    )
    extractor.extract_frames()
