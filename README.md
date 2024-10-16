# project_mht

[English](#english) | [日本語](#japanese)

<a name="english"></a>
## English

### 1. Environment Setup

To set up the GSAM2 environment, follow these steps:

```bash
$ cd {path_to_your_gsam2-folder}
$ pip install -e .
$ pip install --no-build-isolation -e grounding_dino
$ python setup.py build_ext --inplace
```

### 2. Usage

To run the project:

```bash
$ cd {path_to_your_project_mht_folder}
$ source inference.sh
```

### 3. Directory Structure

```
project_mht
    |-gsam2
    |   |-gsam2_c-idv2.py
    |
    |-modules
    |   |-split_save.py
    |   |-correct_id.py
    |   |-merge_segment.py
    |   |-create_correct_id_video.py
    |-data
    |   |-frames
    |   |   |-segment_0
    |   |   |-segment_1
    |   |   |-...
    |   |-gsam2_output
    |   |   |-segment_0
    |   |   |   |-json_data
    |   |   |   |-mask_data
    |   |   |   |-corrected_jsons
    |   |   |-segment_1
    |   |   |-...
    |   |-merged_jsons
    |   |-results.csv
```

### 4. Component Descriptions

#### 4.1 split_save.py

**Purpose**: Extracts and saves image frames from the input folder at specific intervals and durations.

- **Input Arguments**:
  - Input folder path for frames
  - Frame extraction interval (number of frames)
  - Frame extraction duration (number of frames)
  - Output folder path
- **Output**: Segmented frames in the specified output folder.

#### 4.2 gsam2_c-idv2.py

**Purpose**: Processes the frames and generates GSAM2 outputs including mask and JSON data.

- **Input Arguments**:
  - Path to the frames folder
  - Path to the output folder
  - GPU device ID
- **Output**: GSAM2 output folder containing mask data and JSON data.

#### 4.3 correct_id.py

**Purpose**: Combines the initial millisecond-level `_sam2.csv` data with `cc` (connected components) information.

- **Input Arguments**:
  - Path to GSAM2 mask data folder
  - Path to GSAM2 JSON data folder
  - Path to CC information CSV file
  - Path to save the corrected mask data
  - Path to save the corrected JSON data
  - GPU device ID
- **Output**: Corrected JSON files.

#### 4.4 merge_segment.py

**Purpose**: Merges the JSON files from processed segments with connected components (cc) information, ensuring ID continuity across segments.

- **Input Arguments**:
  - Path to the GSAM2 output folder
  - Output folder path for merged JSON files
- **Output**: Merged JSON files stored in the `merged_jsons` folder.

### 5. TODO

- [ ] Update `gsam2_c-id.py` to handle cases when no human is detected.
- [ ] Set the interval for receiving CC information.
- [ ] Implement a loop-closing mechanism to confirm IDs across all CSV files.

<a name="japanese"></a>
## 日本語

### 1. 環境構築

SAM2の環境を構築するには、以下のコマンドを実行してください：

```bash
$ cd {path_to_your_gsam2-foler}
$ pip install -e .
$ pip install --no-build-isolation -e grounding_dino
$ python setup.py build_ext --inplace
```

### 2. 使用方法

プロジェクトを実行するには、以下のコマンドを使用してください：

```bash
$ cd {path_to_your_project_mht_folder}
$ source inference.sh
```

### 3. ディレクトリ構成

```
project_mht
    |-gsam2
    |   |-gsam2_c-idv2.py
    |
    |-modules
    |   |-split_save.py
    |   |-correct_id.py
    |   |-merge_segment.py
    |   |-create_correct_id_video.py
    |-data
    |   |-frames
    |   |   |-segment_0
    |   |   |-segment_1
    |   |   |-...
    |   |-gsam2_output
    |   |   |-segment_0
    |   |   |   |-json_data
    |   |   |   |-mask_data
    |   |   |   |-corrected_jsons
    |   |   |-segment_1
    |   |   |-...
    |   |-merged_jsons
    |   |-results.csv
```

### 4. 構成要素説明

#### 4.1 split_save.py

役割：与えられた切り出し画像ファイル格納のフォルダから任意の間隔とインターバルで切り出して保存する

- 入力引数
  - 入力フレーム格納フォルダパス
  - 任意のインターバル（枚数）
  - 任意のduration（枚数）
  - 出力フォルダパス
- 出力
  - segmentごとのわけられたframesフォルダパス

#### 4.2 gsam2_c-idv2.py

役割：GSam2の処理を実行する

- 入力引数
  - framesフォルダパス
  - 出力フォルダパス
  - gpu device id
- 出力
  - gsam2の出力フォルダ
    - maskデータ
    - jsonデータ

#### 4.3 correct_id.py

役割：はじめのミリ秒_sam2.csvとccをつなぎ合わせる

- 入力引数
  - gsam2のマスクデータフォルダパス
  - gsam2のjsonデータフォルダパス
  - CC情報のcsvファイル
  - CC情報統合したあとのマスクデータフォルダパス
  - CC情報統合したあとのjsonデータフォルダパス
  - gpu device
- 出力
  - 矯正されたjson

#### 4.4 merge_segment.py

役割：segmentに区切られて処理され、ccを紐付けたあとのjsonをid継承をおこないすべて同一フォルダに統合する

- 入力引数
  - gsam2処理のOUTPUTフォルダパス
  - 出力フォルダパス
- 出力
  - マージされたjson（merge_jsonsフォルダ内）

### 5. TODO

- [ ] gsam2_c-id.pyをヒトがうつってないときに対応させる
- [ ] ccをうけとる間隔
- [ ] すべてのcsvでIDを確認するループクロージング的な