#使い方 2026・9・21 usugi
#デスクトップに language フォルダーを作る。
#language フォルダーの下に results フォルダ、images\Naming フォルダーを作る
#Naming フォルダーに 1_りんご.jpg 2_ぶどう.jpg など名前をつけて、画像を入れておく。1_* から読み込むので何枚でも OK
#反応時間は閾値上の音声が入った時間。
#無反応の場合は 15 秒で次にうつる。
#録音は絵が提示した瞬間からエンターが押されるまで保存される。


import numpy as np
import sounddevice as sd
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
import os
import soundfile as sf
import whisper
import sys
import datetime
import json

WHISPER_MODEL = None
THRESHOLD = 0.02
FS = 44100
MAX_DURATION = 15

# 結果フォルダのパス設定

def get_app_dir():
    """
    .pyとして実行した場合：このスクリプトがあるフォルダ
    .exeとして実行した場合：NamingApp.exe があるフォルダ
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


APP_DIR = get_app_dir()

RESULTS_DIR = os.path.join(APP_DIR, "results")
IMG_FOLDER = os.path.join(APP_DIR, "images", "Naming")
MODEL_DIR = os.path.join(APP_DIR, "whisper_model")

# 同じフォルダに置いた ffmpeg.exe を優先して使う
os.environ["PATH"] = APP_DIR + os.pathsep + os.environ.get("PATH", "")

SYNONYMS_JSON_PATH = os.path.join(APP_DIR, "synonyms.json")


DEFAULT_COMMON_SYNONYMS = {
    'みかん': ['オレンジ', '柑子'],
    'オレンジ': ['みかん', '橙'],
    'いちご': ['ストロベリー'],
    'ストロベリー': ['いちご'],
    'ぶどう': ['グレープ'],
    'グレープ': ['ぶどう'],
    'もも': ['ピーチ'],
    'ピーチ': ['もも'],
    'すいか': ['ウォーターメロン'],
    'なし': ['ペアー', '梨'],
    'かき': ['パーシモン', '柿'],
}


def load_common_synonyms():
    """
    exeまたは.pyと同じフォルダにある synonyms.json を読み込む。
    synonyms.jsonが存在しない場合やJSON形式に問題がある場合は、
    上のDEFAULT_COMMON_SYNONYMSを使用する。
    """
    synonyms = DEFAULT_COMMON_SYNONYMS.copy()

    if not os.path.exists(SYNONYMS_JSON_PATH):
        print(f"[注意] synonyms.json が見つかりません。既定の同義語辞書を使用します。")
        print(f"       想定場所: {SYNONYMS_JSON_PATH}")
        return synonyms

    try:
        with open(SYNONYMS_JSON_PATH, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        if not isinstance(loaded_data, dict):
            print("[警告] synonyms.json の最上位は {} で囲んだ辞書形式である必要があります。")
            print("       既定の同義語辞書を使用します。")
            return synonyms

        valid_count = 0

        for answer_word, synonym_list in loaded_data.items():
            if not isinstance(answer_word, str):
                print(f"[警告] 不正な正解語キーを無視しました: {answer_word}")
                continue

            if not isinstance(synonym_list, list):
                print(f"[警告] '{answer_word}' の同義語は [ ] で囲んだリスト形式にしてください。")
                continue

            valid_synonyms = [
                synonym
                for synonym in synonym_list
                if isinstance(synonym, str) and synonym.strip()
            ]

            synonyms[answer_word] = valid_synonyms
            valid_count += 1

        print(f"同義語設定を読み込みました: {SYNONYMS_JSON_PATH}")
        print(f"synonyms.json 内の有効な項目数: {valid_count}")

    except json.JSONDecodeError as e:
        print(f"[警告] synonyms.json の形式が正しくありません: {e}")
        print("       既定の同義語辞書を使用します。")

    except Exception as e:
        print(f"[警告] synonyms.json の読み込み中にエラーが発生しました: {e}")
        print("       既定の同義語辞書を使用します。")

    return synonyms


COMMON_SYNONYMS = load_common_synonyms()




def show_instruction_and_wait(text, font_size=36):
    root = tk.Tk()
    root.title("インストラクション")
    root.attributes('-topmost', True)
    root.attributes('-fullscreen', True)
    root.configure(bg='lightblue')
    
    label = tk.Label(root, text=text, font=("Meiryo", font_size), bg='lightblue', fg='black')
    label.pack(expand=True)
    root.focus_force()
    root.lift()
    
    def close(event=None):
        root.destroy()
    
    root.bind('<Return>', close)
    root.bind('<Escape>', close)
    root.mainloop()
    time.sleep(0.1)


def recognize_from_file(filename):
    global WHISPER_MODEL
    
    result = WHISPER_MODEL.transcribe(
        filename,
        language="ja",
        task="transcribe",
        temperature=0.0,
    )
    
    text = result["text"].strip()
    return text

def play_trial(image_path, trial_num, prompt, output_dir):
    detected_event = threading.Event()
    recording_active = threading.Event()
    stream_ready = threading.Event()

    reaction_time = [None]
    img_shown_time = [None]
    voice_detected = [False]
    audio_buffer = []
    recording_started = [False]
    enter_pressed = [False]
    timed_out = [False]
    audio_error = [None]
    root = [None]

    def audio_monitor():
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[音声入力警告] {status}")

            # 画像提示前の音声は保存しない
            if not recording_active.is_set():
                return

            # 画像提示後からEnterまたはタイムアウトまでを保存する
            audio_buffer.append(indata.copy())

            # 閾値を最初に超えた時刻だけをRTとして記録する
            if not detected_event.is_set():
                vol = np.abs(indata).max()

                if vol > THRESHOLD:
                    dt = time.perf_counter() - img_shown_time[0]

                    reaction_time[0] = dt
                    voice_detected[0] = True
                    recording_started[0] = True
                    detected_event.set()

                    print(f"[音声検出] RT = {dt:.3f} 秒")

        try:
            with sd.InputStream(
                samplerate=FS,
                channels=1,
                callback=callback,
                blocksize=512,
                dtype='float32'
            ):
                # 先にマイク入力ストリームを起動して準備する
                stream_ready.set()

                # 画像提示時刻がセットされるまで待機する
                while img_shown_time[0] is None:
                    time.sleep(0.001)

                start_time = img_shown_time[0]

                # Enterまたは15秒タイムアウトまで待機する
                while (
                    not enter_pressed[0]
                    and time.perf_counter() - start_time < MAX_DURATION
                ):
                    time.sleep(0.005)

                if not enter_pressed[0]:
                    timed_out[0] = True
                    print(f"[タイムアウト] {MAX_DURATION}秒経過")

                    if root[0]:
                        root[0].after(0, lambda: root[0].quit())

        except Exception as e:
            audio_error[0] = e
            stream_ready.set()
            print(f"[音声入力エラー] {e}")

    t_audio = threading.Thread(target=audio_monitor, daemon=True)
    t_audio.start()

    # マイク入力ストリームの準備完了を最大3秒待つ
    if not stream_ready.wait(timeout=3):
        print("[警告] マイク入力ストリームの準備が3秒以内に完了しませんでした。")

    if audio_error[0] is not None:
        print(f"[エラー] マイク入力を開始できませんでした: {audio_error[0]}")
        return None, None, False, False

    root[0] = tk.Tk()





    root[0].title("刺激画像")
    screen_width = root[0].winfo_screenwidth()
    screen_height = root[0].winfo_screenheight()
    root[0].geometry(f"{screen_width}x{screen_height}")
    root[0].attributes('-fullscreen', True)
    root[0].attributes('-topmost', True)
    root[0].configure(bg='white')
    root[0].focus_force()
    root[0].lift()
    root[0].focus_set()
    
    try:
        img = Image.open(image_path)
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(root[0], image=tk_img, bg='white')
        label.image = tk_img
        label.pack(expand=True, fill=tk.BOTH)
    except Exception as e:
        root[0].destroy()
        return None, None, False, False
    
    # Labelのサイズ・配置などの描画処理を反映する
    root[0].update_idletasks()

    # 画像提示時刻を記録する
    img_shown_time[0] = time.perf_counter()

    # この瞬間以後のマイク入力を録音データへ追加する
    recording_active.set()

    print("[刺激提示] 録音・RT計測を開始しました")

    
    def on_enter(event=None):
        print("[Enter 検知] 録音停止、次へ")
        enter_pressed[0] = True
        root[0].quit()
        root[0].destroy()
    
    root[0].bind('<Return>', on_enter)
    root[0].bind('<Escape>', lambda e: on_enter())
    root[0].bind('<space>', lambda e: on_enter())
    
    root[0].after(10, lambda: root[0].focus_force())
    root[0].after(50, lambda: root[0].lift())
    root[0].after(100, lambda: root[0].focus_set())
    
    root[0].mainloop()
    t_audio.join()

    saved_file = None
    if audio_buffer and voice_detected[0]:
        recorded_audio = np.concatenate(audio_buffer, axis=0)
        saved_file = os.path.join(output_dir, f"voice_{trial_num:02d}_{prompt}.wav")
        sf.write(saved_file, recorded_audio, FS)
        print(f"[保存完了] {saved_file} ({len(recorded_audio)/FS:.2f}秒)")
    elif audio_buffer and timed_out:
        recorded_audio = np.concatenate(audio_buffer, axis=0)
        saved_file = os.path.join(output_dir, f"voice_{trial_num:02d}_{prompt}_timeout.wav")
        sf.write(saved_file, recorded_audio, FS)
        print(f"[タイムアウト保存] {saved_file} ({len(recorded_audio)/FS:.2f}秒)")
    
    return reaction_time[0], saved_file, timed_out[0], False


def get_image_files(folder_path):
    files = []
    if not os.path.exists(folder_path):
        return files
    
    for f in os.listdir(folder_path):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')) or '_' not in f:
            continue
        parts = f.split('_', 1)
        if len(parts) != 2:
            continue
        num_str, rest_with_ext = parts
        if not num_str.isdigit():
            continue
        answer_word = None
        for ext in ['.jpg', '.jpeg', '.png']:
            if rest_with_ext.lower().endswith(ext):
                answer_word = rest_with_ext[:-len(ext)]
                break
        if answer_word is None:
            continue
        files.append((int(num_str), f, answer_word))
    
    files.sort(key=lambda x: x[0])
    return files


def to_hiragana(text):
    hiragana_map = str.maketrans({
        'ア': 'あ', 'イ': 'い', 'ウ': 'う', 'エ': 'え', 'オ': 'お',
        'カ': 'か', 'キ': 'き', 'ク': 'く', 'ケ': 'け', 'コ': 'こ',
        'サ': 'さ', 'シ': 'し', 'ス': 'す', 'セ': 'せ', 'ソ': 'そ',
        'タ': 'た', 'チ': 'ち', 'ツ': 'つ', 'テ': 'て', 'ト': 'と',
        'ナ': 'な', 'ニ': 'に', 'ヌ': 'ぬ', 'ネ': 'ね', 'ノ': 'の',
        'ハ': 'は', 'ヒ': 'ひ', 'フ': 'ふ', 'ヘ': 'へ', 'ホ': 'ほ',
        'マ': 'ま', 'ミ': 'み', 'ム': 'む', 'メ': 'め', 'モ': 'も',
        'ヤ': 'や', 'ユ': 'ゆ', 'ヨ': 'よ',
        'ラ': 'ら', 'リ': 'り', 'ル': 'る', 'レ': 'れ', 'ロ': 'ろ',
        'ワ': 'わ', 'ヲ': 'を', 'ン': 'ん',
        'ガ': 'が', 'ギ': 'ぎ', 'グ': 'ぐ', 'ゲ': 'げ', 'ゴ': 'ご',
        'ザ': 'ざ', 'ジ': 'じ', 'ズ': 'ず', 'ゼ': 'ぜ', 'ゾ': 'ぞ',
        'ダ': 'だ', 'ヂ': 'ぢ', 'ヅ': 'づ', 'デ': 'で', 'ド': 'ど',
        'バ': 'ば', 'ビ': 'び', 'ブ': 'ぶ', 'ベ': 'べ', 'ボ': 'ぼ',
        'パ': 'ぱ', 'ピ': 'ぴ', 'プ': 'ぷ', 'ペ': 'ぺ', 'ポ': 'ぽ',
    })
    return text.translate(hiragana_map)


def check_answer(recognized_text, answer_word):
    if not recognized_text:
        return False
    
    if answer_word in recognized_text:
        return True
    
    recognized_hira = to_hiragana(recognized_text)
    answer_hira = to_hiragana(answer_word)
    
    if answer_hira in recognized_hira:
        return True
    
    if answer_word in COMMON_SYNONYMS:
        for synonym in COMMON_SYNONYMS[answer_word]:
            if synonym in recognized_text:
                return True
            synonym_hira = to_hiragana(synonym)
            if synonym_hira in recognized_hira:
                return True
    
    return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法：python script_naming_NEW.py [ID] [名前]")
        sys.exit(1)
    
    SUBJECT_ID = sys.argv[1]
    SUBJECT_NAME = sys.argv[2] if len(sys.argv) >= 3 else ""
    
    # results フォルダを作成（既存の場合はそのまま）
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"results フォルダを作成しました：{RESULTS_DIR}")
    else:
        print(f"results フォルダを使用します：{RESULTS_DIR}")
    
    # 被験者フォルダ（results/ID_名前/）
    if SUBJECT_NAME:
        SUBJECT_FOLDER = f"{SUBJECT_ID}_{SUBJECT_NAME}"
    else:
        SUBJECT_FOLDER = f"{SUBJECT_ID}"
    
    SUBJECT_DIR = os.path.join(RESULTS_DIR, SUBJECT_FOLDER)
    os.makedirs(SUBJECT_DIR, exist_ok=True)
    print(f"被験者フォルダ：{SUBJECT_DIR}")
    
    # 出力ディレクトリ（results/ID_名前/Naming_日時_ID_名前/）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if SUBJECT_NAME:
        FILE_PREFIX = f"Naming_{timestamp}_{SUBJECT_ID}_{SUBJECT_NAME}"
    else:
        FILE_PREFIX = f"Naming_{timestamp}_{SUBJECT_ID}"
    
    OUTPUT_DIR = os.path.join(SUBJECT_DIR, FILE_PREFIX)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"出力ディレクトリ：{OUTPUT_DIR}")
    
    image_files = get_image_files(IMG_FOLDER)
    if not image_files:
        print("画像が見つかりませんでした。")
        sys.exit(1)
    
    print(f"画像ファイル数：{len(image_files)}")
    
    # large モデル（最高精度）
    print("Whisper large モデルをロード中...（初回のみ 3-5 分かかります）")
    WHISPER_MODEL = whisper.load_model("large", download_root=MODEL_DIR)
    print("完了\n")
    
    results = []
    correct_count = 0

    for idx, (trial_num, filename, answer_word) in enumerate(image_files, 1):
        img_path = os.path.join(IMG_FOLDER, filename)
        print(f"試行 {idx}/{len(image_files)}: {answer_word}")
        
        show_instruction_and_wait(f'Enter キーを押してください\n\n（試行 {idx}/{len(image_files)}）', font_size=48)
        
        if not os.path.exists(img_path):
            results.append(f"{trial_num}\t{answer_word}\t---\t画像なし\t---\t---")
            continue
        
        rt, fname, timed_out, _ = play_trial(img_path, trial_num, answer_word, OUTPUT_DIR)
        
        result_str = "反応なし"
        recog_text = ""
        
        if timed_out:
            print(f"[タイムアウト] 音声検出なし、次の問題へ")
            results.append(f"{trial_num}\t{answer_word}\t---\t反応なし\t---\t---")
        elif rt is not None and fname is not None:
            print(f"音声認識中...")
            recognized_text = recognize_from_file(fname)
            print(f"認識結果：{recognized_text}")
            recog_text = recognized_text if recognized_text else "---"
            
            if check_answer(recognized_text, answer_word):
                result_str = "正解"
                correct_count += 1
                print(f"→ 正解！")
            else:
                result_str = "不正解"
                print(f"→ 不正解（正解語：{answer_word}）")
            
            results.append(f"{trial_num}\t{answer_word}\t{rt:.3f}\t{result_str}\t{fname}\t{recog_text}")
        else:
            results.append(f"{trial_num}\t{answer_word}\t---\t反応なし\t---\t---")

    total = len(image_files)
    accuracy = correct_count / total * 100 if total > 0 else 0

    result_txt = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_naming_results.txt")
    
    # 結果ファイルの保存
    print(f"\n結果を保存中... {result_txt}")
    try:
        with open(result_txt, "w", encoding="utf-8") as f:
            f.write("試行番号\t正解語\t反応時間（秒）\t正誤\t録音ファイル\t認識結果\n")
            for line in results:
                f.write(line + "\n")
            f.write(f"\n総試行数：{total}\n")
            f.write(f"正解数：{correct_count}\n")
            f.write(f"正答率：{accuracy:.1f}%\n")
            f.write(f"\n被験者 ID: {SUBJECT_ID}\n")
            if SUBJECT_NAME:
                f.write(f"被験者名：{SUBJECT_NAME}\n")
            f.write(f"測定日時：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[保存完了] {result_txt}")
    except Exception as e:
        print(f"[エラー] 結果の保存に失敗しました：{e}")
        print(f"保存先：{result_txt}")

    print(f"\n実験完了！結果を{result_txt}に保存しました。")
