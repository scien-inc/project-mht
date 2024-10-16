import os
import json
import argparse
import shutil
import torch
from torchvision.ops import box_iou

def unify_instance_ids(base_dir, merge_dir):
    """
    各セグメントフォルダ内のJSONファイルの 'instance_id' を統一し、順番にマージしていきます。
    同じファイル名が存在しない場合は、無条件で 'merge_dir' にコピーします。

    :param base_dir: セグメントフォルダが存在するベースディレクトリ
    :param merge_dir: マージ結果を保存するディレクトリ
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # セグメントフォルダの一覧を取得し、フォルダ名でソート
    segment_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    segment_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # 'segment_0', 'segment_1', ...

    # 各セグメントのファイル名の集合を取得
    segment_files = {}
    for idx, segment_dir in enumerate(segment_dirs):
        corrected_jsons_path = os.path.join(base_dir, segment_dir, 'corrected_jsons')
        if not os.path.exists(corrected_jsons_path):
            print(f"'corrected_jsons' ディレクトリが見つかりません: {corrected_jsons_path}")
            continue
        files = set(os.listdir(corrected_jsons_path))
        segment_files[idx] = files

    # すべてのファイル名の集合を取得
    all_files = set()
    for files in segment_files.values():
        all_files.update(files)

    # ファイル名ごとに、存在するセグメントのリストを作成
    file_segments = {}  # {filename: [segment indices]}
    for filename in all_files:
        file_segments[filename] = []
        for idx, files in segment_files.items():
            if filename in files:
                file_segments[filename].append(idx)

    os.makedirs(merge_dir, exist_ok=True)

    # インスタンスIDのマッピングを初期化
    instance_id_mapping = {}  # {('segment_index', old_instance_id): unified_instance_id}

    # 各ファイルを処理
    for filename in all_files:
        segments_with_file = file_segments[filename]
        # ファイルが複数のセグメントに存在する場合
        if len(segments_with_file) > 1:
            # 各セグメントからデータを読み込み
            data_segments = {}
            labels_segments = {}
            bboxes_segments = {}
            instance_ids_segments = {}
            class_names_segments = {}
            label_keys_segments = {}  # 修正：ラベルキーを保持
            for seg_idx in segments_with_file:
                segment_dir = segment_dirs[seg_idx]
                corrected_jsons_path = os.path.join(base_dir, segment_dir, 'corrected_jsons')
                with open(os.path.join(corrected_jsons_path, filename), 'r') as f:
                    data = json.load(f)
                    data_segments[seg_idx] = data
                    labels = data.get('labels', {})
                    labels_segments[seg_idx] = labels
                    bboxes = []
                    instance_ids = []
                    class_names = []
                    label_keys = []  # 修正：ラベルキーを保持
                    for key, label in labels.items():
                        bbox = [label['x1'], label['y1'], label['x2'], label['y2']]
                        bboxes.append(bbox)
                        instance_ids.append(label['instance_id'])
                        class_names.append(label['class_name'])
                        label_keys.append(key)  # 修正：ラベルキーを追加
                    bboxes_segments[seg_idx] = bboxes
                    instance_ids_segments[seg_idx] = instance_ids
                    class_names_segments[seg_idx] = class_names
                    label_keys_segments[seg_idx] = label_keys  # 修正：ラベルキーを保存

            # 最初のセグメントを基準にする
            base_seg_idx = segments_with_file[0]
            base_bboxes = bboxes_segments[base_seg_idx]
            base_instance_ids = instance_ids_segments[base_seg_idx]
            base_class_names = class_names_segments[base_seg_idx]
            base_label_keys = label_keys_segments[base_seg_idx]  # 修正：ラベルキーを取得

            # バウンディングボックスをテンソルに変換
            base_bboxes_tensor = torch.tensor(base_bboxes, dtype=torch.float32, device=device)

            # 他のセグメントと比較
            for other_seg_idx in segments_with_file[1:]:
                other_bboxes = bboxes_segments[other_seg_idx]
                other_instance_ids = instance_ids_segments[other_seg_idx]
                other_class_names = class_names_segments[other_seg_idx]
                other_label_keys = label_keys_segments[other_seg_idx]  # 修正：ラベルキーを取得

                other_bboxes_tensor = torch.tensor(other_bboxes, dtype=torch.float32, device=device)

                # IoUを計算
                if base_bboxes_tensor.size(0) > 0 and other_bboxes_tensor.size(0) > 0:
                    ious = box_iou(base_bboxes_tensor, other_bboxes_tensor)
                    iou_threshold = 0.9
                    matching_pairs = torch.nonzero(ious >= iou_threshold, as_tuple=False)

                    for idx_pair in matching_pairs:
                        base_idx = idx_pair[0].item()
                        other_idx = idx_pair[1].item()
                        if base_class_names[base_idx] == other_class_names[other_idx]:
                            base_id = base_instance_ids[base_idx]
                            other_id = other_instance_ids[other_idx]

                            base_label_key = base_label_keys[base_idx]  # 修正：ラベルキーを使用
                            other_label_key = other_label_keys[other_idx]  # 修正：ラベルキーを使用

                            # former_instance_idを追加
                            labels_segments[base_seg_idx][base_label_key]['former_instance_id'] = base_id
                            labels_segments[other_seg_idx][other_label_key]['former_instance_id'] = other_id

                            # 統一されたinstance_idを決定（基準のセグメントのIDを使用）
                            unified_id = base_id

                            # マッピングを記録
                            instance_id_mapping[(other_seg_idx, other_id)] = unified_id
                            instance_id_mapping[(base_seg_idx, base_id)] = unified_id

                            # instance_idを更新
                            labels_segments[base_seg_idx][base_label_key]['instance_id'] = unified_id
                            labels_segments[other_seg_idx][other_label_key]['instance_id'] = unified_id

            # マージしたラベルを作成
            merged_labels = {}
            label_counter = 1
            for seg_idx in segments_with_file:
                labels = labels_segments[seg_idx]
                for key in labels:
                    label = labels[key]
                    if 'former_instance_id' not in label:
                        label['former_instance_id'] = label['instance_id']
                        key_mapping = (seg_idx, label['instance_id'])
                        label['instance_id'] = instance_id_mapping.get(key_mapping, label['instance_id'])
                    merged_labels[str(label_counter)] = label
                    label_counter +=1

            # マージされたデータを保存
            merged_data = data_segments[base_seg_idx]
            merged_data['labels'] = merged_labels
            with open(os.path.join(merge_dir, filename), 'w') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=4)

        else:
            # ファイルが一つのセグメントにのみ存在する場合
            seg_idx = segments_with_file[0]
            segment_dir = segment_dirs[seg_idx]
            corrected_jsons_path = os.path.join(base_dir, segment_dir, 'corrected_jsons')
            src_file = os.path.join(corrected_jsons_path, filename)
            dst_file = os.path.join(merge_dir, filename)
            shutil.copy2(src_file, dst_file)

            # データを読み込み
            with open(dst_file, 'r') as f:
                data = json.load(f)
            labels = data.get('labels', {})
            bboxes = []
            instance_ids = []
            class_names = []
            label_keys = []
            for key, label in labels.items():
                bbox = [label['x1'], label['y1'], label['x2'], label['y2']]
                bboxes.append(bbox)
                instance_ids.append(label['instance_id'])
                class_names.append(label['class_name'])
                label_keys.append(key)

            # バウンディングボックスをテンソルに変換
            if bboxes:
                bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32, device=device)
                # 同じファイル内でIoUを計算
                ious = box_iou(bboxes_tensor, bboxes_tensor)
                iou_threshold = 0.9
                num_labels = len(bboxes)
                for i in range(num_labels):
                    label_i_key = label_keys[i]
                    label_i = labels[label_i_key]
                    label_i['former_instance_id'] = label_i.get('former_instance_id', label_i['instance_id'])
                    old_id_i = label_i['former_instance_id']
                    for j in range(i+1, num_labels):
                        if ious[i, j] >= iou_threshold and class_names[i] == class_names[j]:
                            label_j_key = label_keys[j]
                            label_j = labels[label_j_key]
                            label_j['former_instance_id'] = label_j.get('former_instance_id', label_j['instance_id'])
                            old_id_j = label_j['former_instance_id']
                            if old_id_i == old_id_j:
                                # instance_idを統一
                                key_i = (seg_idx, old_id_i)
                                unified_id = instance_id_mapping.get(key_i, label_i['instance_id'])
                                label_i['instance_id'] = unified_id
                                label_j['instance_id'] = unified_id
                                # マッピングを更新
                                instance_id_mapping[(seg_idx, old_id_i)] = unified_id
                                instance_id_mapping[(seg_idx, old_id_j)] = unified_id

            # instance_idをグローバルマッピングで更新
            updated = False
            for key in labels:
                label = labels[key]
                old_id = label['instance_id']
                label['former_instance_id'] = label.get('former_instance_id', old_id)
                key_mapping = (seg_idx, label['former_instance_id'])
                if key_mapping in instance_id_mapping:
                    unified_id = instance_id_mapping[key_mapping]
                    if unified_id != old_id:
                        label['instance_id'] = unified_id
                        updated = True

            # 更新されたデータを保存
            with open(dst_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    # マージディレクトリ内のすべてのファイルの instance_id を最終更新
    for json_file in os.listdir(merge_dir):
        file_path = os.path.join(merge_dir, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        labels = data.get('labels', {})
        updated = False
        for key in labels:
            label = labels[key]
            old_id = label['instance_id']
            label['former_instance_id'] = label.get('former_instance_id', old_id)
            # 全セグメントを探索
            for seg_idx in range(len(segment_dirs)):
                key_mapping = (seg_idx, label['former_instance_id'])
                if key_mapping in instance_id_mapping:
                    unified_id = instance_id_mapping[key_mapping]
                    if unified_id != old_id:
                        label['instance_id'] = unified_id
                        updated = True
                        break
        if updated:
            with open(file_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Instance ID Unifier')
    parser.add_argument('--base_dir', type=str, required=True, help='セグメントフォルダが存在するベースディレクトリのパス')
    parser.add_argument('--merge_dir', type=str, required=True, help='マージ結果を保存するディレクトリのパス')
    args = parser.parse_args()

    unify_instance_ids(args.base_dir, args.merge_dir)

if __name__ == "__main__":
    main()
