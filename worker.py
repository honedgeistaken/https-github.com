import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import yt_dlp
import requests
from supabase import create_client
from faster_whisper import WhisperModel
from youtube_transcript_api import YouTubeTranscriptApi


SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

BASE_TMP_DIR = Path("/tmp/social_video_worker")
BASE_TMP_DIR.mkdir(parents=True, exist_ok=True)

_WHISPER_MODEL = None


def now_iso():
    return datetime.utcnow().isoformat()


def get_db():
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL secret eksik.")
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY secret eksik.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def log_event(job_id, level, action, message, details=""):
    try:
        db = get_db()
        db.table("analyzer_logs").insert({
            "job_id": job_id,
            "level": level,
            "action": action,
            "message": message,
            "details": details,
            "created_at": now_iso()
        }).execute()
    except Exception:
        pass


def safe_int(value, fallback):
    try:
        return int(float(value))
    except Exception:
        return fallback


def get_settings():
    defaults = {
        "MAX_VIDEO_HEIGHT": "480",
        "MAX_VIDEO_MB": "200",
        "MAX_VIDEO_DURATION_SECONDS": "600",
        "MAX_FRAMES_PER_VIDEO": "12",
        "AUTO_DELETE_TEMP": "TRUE",
        "FREE_MODE": "TRUE"
    }

    try:
        db = get_db()
        response = db.table("analyzer_settings").select("key,value").execute()
        rows = response.data or []

        for row in rows:
            defaults[str(row["key"])] = str(row["value"])

    except Exception:
        pass

    return defaults


def get_next_queued_job():
    db = get_db()

    response = (
        db.table("video_jobs")
        .select("*")
        .eq("status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )

    jobs = response.data or []

    if not jobs:
        return None

    return jobs[0]


def update_job(job_id, updates):
    db = get_db()
    db.table("video_jobs").update(updates).eq("id", job_id).execute()


def get_whisper_model():
    global _WHISPER_MODEL

    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8"
        )

    return _WHISPER_MODEL


def supported_single_video_link(link_type):
    allowed = {
        "youtube_video",
        "youtube_shorts",
        "instagram_reel",
        "instagram_post",
        "tiktok_video",
        "tiktok_short_link"
    }

    return link_type in allowed


def find_downloaded_video(work_dir):
    candidates = []

    for ext in ["*.mp4", "*.webm", "*.mkv", "*.mov"]:
        candidates.extend(work_dir.glob(ext))

    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        raise RuntimeError("Video dosyası bulunamadı. Platform indirmeye izin vermemiş olabilir.")

    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)

    return candidates[0]


def download_video(job, work_dir, settings):
    url = job.get("url", "")
    link_type = job.get("link_type", "")

    if not supported_single_video_link(link_type):
        raise RuntimeError(
            "Bu worker şu an sadece tek video / Reels / Shorts linklerini işler. "
            "Profil ve kanal toplu analizini sonraki aşamada ekleyeceğiz."
        )

    max_height = safe_int(settings.get("MAX_VIDEO_HEIGHT"), 480)
    max_mb = safe_int(settings.get("MAX_VIDEO_MB"), 200)
    max_duration = safe_int(settings.get("MAX_VIDEO_DURATION_SECONDS"), 600)

    common_opts = {
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "socket_timeout": 60,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 5,
        "force_ipv4": True,
        "http_chunk_size": 10 * 1024 * 1024,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,tr;q=0.8"
        }
    }

    client_attempts = [
        ["android"],
        ["web"],
        ["ios"],
        ["mweb"]
    ]

    last_error = None
    info = None

    for clients in client_attempts:
        try:
            info_opts = dict(common_opts)
            info_opts.update({
                "extractor_args": {
                    "youtube": {
                        "player_client": clients
                    }
                },
                "impersonate": "chrome"
            })

            with yt_dlp.YoutubeDL(info_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            break

        except Exception as e:
            last_error = e
            continue

    if info is None:
        raise RuntimeError(
            "Video bilgisi alınamadı. Platform erişimi sorun çıkardı. "
            f"Son hata: {str(last_error)}"
        )

    duration = info.get("duration")

    if duration and duration > max_duration:
        raise RuntimeError(
            f"Video süresi çok uzun: {duration} sn. Limit: {max_duration} sn."
        )

    title = info.get("title") or ""
    uploader = info.get("uploader") or info.get("channel") or ""
    description = info.get("description") or ""

    download_last_error = None

    for clients in client_attempts:
        try:
            ydl_opts = dict(common_opts)
            ydl_opts.update({
                "outtmpl": str(work_dir / "video.%(ext)s"),
                "format": (
                    f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
                    f"b[height<={max_height}][ext=mp4]/"
                    f"best[height<={max_height}]/best"
                ),
                "merge_output_format": "mp4",
                "max_filesize": max_mb * 1024 * 1024,
                "extractor_args": {
                    "youtube": {
                        "player_client": clients
                    }
                },
                "impersonate": "chrome"
            })

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)

            video_path = find_downloaded_video(work_dir)

            size_mb = video_path.stat().st_size / (1024 * 1024)

            if size_mb > max_mb:
                raise RuntimeError(
                    f"Video dosyası çok büyük: {size_mb:.1f} MB. Limit: {max_mb} MB."
                )

            return {
                "video_path": video_path,
                "title": title,
                "uploader": uploader,
                "description": description,
                "duration": duration,
                "size_mb": round(size_mb, 2)
            }

        except Exception as e:
            download_last_error = e
            continue

    raise RuntimeError(
        "Video indirilemedi. Platform engeli, geçici bağlantı sorunu veya erişim kısıtı olabilir. "
        f"Son hata: {str(download_last_error)}"
    )


def extract_audio(video_path, audio_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(audio_path)
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if completed.returncode != 0:
        raise RuntimeError("FFmpeg ses çıkarma hatası: " + completed.stderr[-1000:])

    if not audio_path.exists():
        raise RuntimeError("Ses dosyası oluşturulamadı.")

    return audio_path


def extract_strong_lines(text):
    if not text:
        return ""

    pieces = re.split(r"(?<=[.!?])\s+", text)
    pieces = [p.strip() for p in pieces if len(p.strip()) >= 35]

    question_lines = [p for p in pieces if "?" in p]
    long_lines = sorted(pieces, key=len, reverse=True)

    selected = []

    for item in question_lines + long_lines:
        if item not in selected:
            selected.append(item)

        if len(selected) >= 5:
            break

    return "\n".join(selected)


def detect_cta_text(text):
    if not text:
        return ""

    keywords = [
        "takip", "yorum", "kaydet", "paylaş", "abone",
        "subscribe", "follow", "comment", "save", "share",
        "like", "link", "dm"
    ]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    found = []

    for sentence in sentences:
        low = sentence.lower()

        if any(k in low for k in keywords):
            found.append(sentence.strip())

    return "\n".join(found[:5])


def transcribe_audio(audio_path):
    model = get_whisper_model()

    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,
        beam_size=1
    )

    full_parts = []
    timed_parts = []
    first_3_parts = []

    for seg in segments:
        text = (seg.text or "").strip()

        if not text:
            continue

        full_parts.append(text)

        timed_parts.append({
            "start": round(float(seg.start), 2),
            "end": round(float(seg.end), 2),
            "text": text
        })

        if float(seg.start) <= 3:
            first_3_parts.append(text)

    full_transcript = " ".join(full_parts).strip()

    if len(full_transcript) > 60000:
        full_transcript = full_transcript[:60000] + "\n\n[Transkript uzun olduğu için kesildi.]"

    return {
        "language": getattr(info, "language", "") or "",
        "full_transcript": full_transcript,
        "timed_transcript": timed_parts[:200],
        "first_3_seconds_text": " ".join(first_3_parts).strip(),
        "strong_lines": extract_strong_lines(full_transcript),
        "weak_lines": "",
        "cta_text": detect_cta_text(full_transcript)
    }


def score_scene(brightness, contrast, sharpness):
    score = 50

    if 70 <= brightness <= 185:
        score += 15

    if contrast >= 35:
        score += 15

    if sharpness >= 120:
        score += 20

    return max(0, min(100, int(score)))


def extract_frames(video_path, work_dir, max_frames):
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Video OpenCV ile açılamadı.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps > 0 else 0

    if duration <= 0:
        duration = 1

    scenes = []

    for i in range(max_frames):
        target_second = (duration / max_frames) * i + 0.5
        cap.set(cv2.CAP_PROP_POS_MSEC, target_second * 1000)

        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        height, width = frame.shape[:2]
        target_width = 640

        if width > target_width:
            ratio = target_width / width
            new_height = int(height * ratio)
            frame = cv2.resize(frame, (target_width, new_height))

        frame_name = f"frame_{i + 1:03d}.jpg"
        frame_path = frames_dir / frame_name

        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if brightness < 70:
            light_note = "karanlık"
        elif brightness > 185:
            light_note = "çok parlak"
        else:
            light_note = "dengeli ışık"

        if sharpness < 80:
            sharpness_note = "düşük netlik"
        elif sharpness > 250:
            sharpness_note = "yüksek netlik"
        else:
            sharpness_note = "orta netlik"

        scenes.append({
            "scene_no": i + 1,
            "start_second": round(float(target_second), 2),
            "end_second": round(float(target_second + 1), 2),
            "frame_name": frame_name,
            "visual_description": f"Otomatik frame analizi: {light_note}, {sharpness_note}.",
            "screen_text": "OCR bu sürümde ekli değil.",
            "camera_angle": "Otomatik açı tespiti bu sürümde sınırlı.",
            "person_face": "Yüz/kişi analizi bu sürümde yapılmadı.",
            "location": "Otomatik lokasyon tespiti bu sürümde yapılmadı.",
            "emotion": "Duygu analizi bu sürümde yapılmadı.",
            "edit_note": f"Parlaklık: {brightness:.1f}, kontrast: {contrast:.1f}, netlik: {sharpness:.1f}",
            "social_media_note": "Gelişmiş görsel yorum sonraki aşamada eklenecek.",
            "scene_score": score_scene(brightness, contrast, sharpness)
        })

    cap.release()

    return scenes


def score_transcript(transcript):
    if not transcript:
        return 0

    words = transcript.split()
    score = 40

    if len(words) >= 30:
        score += 20

    if len(words) >= 80:
        score += 20

    if "?" in transcript:
        score += 10

    if detect_cta_text(transcript):
        score += 10

    return max(0, min(100, score))


def score_hook(first_3):
    if not first_3:
        return 0

    score = 40

    if len(first_3) >= 12:
        score += 20

    if "?" in first_3:
        score += 20

    strong_words = [
        "neden", "nasıl", "bunu", "hata", "sakın", "bilmen",
        "secret", "mistake", "how", "why"
    ]

    low = first_3.lower()

    if any(w in low for w in strong_words):
        score += 20

    return max(0, min(100, score))


def get_content_pillar(platform, transcript):
    low = (transcript or "").lower()

    if any(k in low for k in ["nasıl", "how", "ipucu", "öğren", "bilmen", "tips"]):
        return "Eğitim / Rehber"

    if any(k in low for k in ["hata", "yanlış", "mistake", "dikkat"]):
        return "Hata / Mit Kırma"

    if any(k in low for k in ["ben", "biz", "hikaye", "story", "deneyim"]):
        return "Deneyim / Storytelling"

    if platform == "youtube":
        return "YouTube Video Analizi"

    if platform == "instagram":
        return "Instagram Reels Analizi"

    if platform == "tiktok":
        return "TikTok Video Analizi"

    return "Genel Video Analizi"


def create_improved_hook(first_3, platform):
    if first_3:
        return f"Bu videodaki ana fikri daha güçlü açmak için: “{first_3[:120]}...” cümlesini daha kısa ve merak uyandıran hale getir."

    if platform == "youtube":
        return "Bu videoyu izlemeye başlamadan önce şunu bilmen gerekiyor..."

    if platform == "instagram":
        return "Bu Reels’i geçmeden önce şu detaya dikkat et..."

    if platform == "tiktok":
        return "Bunu çoğu kişi yanlış yapıyor..."

    return "Bu videoda bilmen gereken en önemli şey şu..."


def create_improved_caption(platform):
    if platform == "youtube":
        return "Bu videodan aldığın en net fikri yorumlara yaz. Devamı için kanalı takip et."

    if platform == "instagram":
        return "Bunu sonra kullanmak için kaydet. Devamı için takipte kal."

    if platform == "tiktok":
        return "Sen olsan bunu nasıl yapardın? Yoruma yaz."

    return "Fikrini yorumlara yaz ve devamı için takip et."


def build_result(job, download_meta, transcript_data, scenes):
    transcript = transcript_data.get("full_transcript", "") if transcript_data else ""
    first_3 = transcript_data.get("first_3_seconds_text", "") if transcript_data else ""
    cta_text = transcript_data.get("cta_text", "") if transcript_data else ""

    visual_score = 0

    if scenes:
        visual_score = int(sum(s["scene_score"] for s in scenes) / len(scenes))

    transcript_score = score_transcript(transcript)
    hook_score = score_hook(first_3)
    cta_score = 80 if cta_text else 35 if transcript else 0

    avg_score = int((visual_score + transcript_score + hook_score + cta_score) / 4)

    if avg_score >= 75:
        viral_potential = "yüksek"
    elif avg_score >= 50:
        viral_potential = "orta"
    else:
        viral_potential = "düşük"

    return {
        "job_id": job["id"],
        "video_url": job.get("url"),
        "platform": job.get("platform"),
        "link_type": job.get("link_type"),
        "title": download_meta.get("title") or "",
        "caption": (download_meta.get("description") or "")[:5000],
        "duration_seconds": int(download_meta.get("duration") or 0) if download_meta.get("duration") else None,
        "uploader": download_meta.get("uploader") or "",
        "transcript_status": "completed" if transcript else "empty_or_disabled",
        "visual_status": "completed" if scenes else "empty_or_disabled",
        "content_pillar": get_content_pillar(job.get("platform"), transcript),
        "hook_type": "Soru/merak hook" if "?" in first_3 else "Genel açılış",
        "first_3_seconds_analysis": first_3 or "İlk 3 saniye metni çıkarılamadı.",
        "main_message": transcript[:1000] if transcript else "Transkript çıkarılamadı veya kapalıydı.",
        "cta_analysis": cta_text or "Belirgin CTA tespit edilmedi.",
        "edit_style": "Temel frame analizi yapıldı. Gelişmiş vision yorumu sonraki aşamada eklenecek.",
        "visual_score": visual_score,
        "transcript_score": transcript_score,
        "hook_score": hook_score,
        "cta_score": cta_score,
        "viral_potential": viral_potential,
        "strengths": "Video başarıyla indirildi, işlendi ve geçici dosya temizliği uygulandı.",
        "weaknesses": "Bu ücretsiz sürümde görsel analiz temel metriklerle sınırlıdır.",
        "improved_hook": create_improved_hook(first_3, job.get("platform")),
        "improved_caption": create_improved_caption(job.get("platform")),
        "recommendations": "\n".join([
            "İlk 3 saniyeyi daha net bir soru veya iddia ile güçlendir.",
            "Ekran yazısını kısa, büyük ve okunur tut.",
            "Videonun sonunda tek bir doğal CTA kullan.",
            "Kaydetme/paylaşma sebebi oluşturacak somut bir bilgi ekle."
        ]),
        "raw_json": {
            "mode": "github_actions_worker_v1",
            "download": {
                "title": download_meta.get("title"),
                "uploader": download_meta.get("uploader"),
                "duration": download_meta.get("duration"),
                "size_mb": download_meta.get("size_mb")
            },
            "scores": {
                "visual_score": visual_score,
                "transcript_score": transcript_score,
                "hook_score": hook_score,
                "cta_score": cta_score,
                "avg_score": avg_score
            },
            "temp_files_deleted": True
        },
        "created_at": now_iso()
    }


def insert_transcript(job, transcript_data):
    if not transcript_data:
        return

    db = get_db()

    db.table("video_transcripts").insert({
        "job_id": job["id"],
        "video_url": job.get("url"),
        "language": transcript_data.get("language"),
        "full_transcript": transcript_data.get("full_transcript"),
        "first_3_seconds_text": transcript_data.get("first_3_seconds_text"),
        "strong_lines": transcript_data.get("strong_lines"),
        "weak_lines": transcript_data.get("weak_lines"),
        "cta_text": transcript_data.get("cta_text"),
        "notes": json.dumps({
            "timed_transcript_preview": transcript_data.get("timed_transcript", [])[:20]
        }, ensure_ascii=False),
        "created_at": now_iso()
    }).execute()


def insert_scenes(job, scenes):
    if not scenes:
        return

    db = get_db()
    rows = []

    for scene in scenes:
        rows.append({
            "job_id": job["id"],
            "video_url": job.get("url"),
            "scene_no": scene.get("scene_no"),
            "start_second": scene.get("start_second"),
            "end_second": scene.get("end_second"),
            "frame_name": scene.get("frame_name"),
            "visual_description": scene.get("visual_description"),
            "screen_text": scene.get("screen_text"),
            "camera_angle": scene.get("camera_angle"),
            "person_face": scene.get("person_face"),
            "location": scene.get("location"),
            "emotion": scene.get("emotion"),
            "edit_note": scene.get("edit_note"),
            "social_media_note": scene.get("social_media_note"),
            "scene_score": scene.get("scene_score"),
            "created_at": now_iso()
        })

    db.table("video_scenes").insert(rows).execute()


def insert_result(result):
    db = get_db()
    db.table("video_results").insert(result).execute()


def youtube_oembed_metadata(url):
    try:
        response = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=20
        )

        if response.status_code != 200:
            return {}

        data = response.json()

        return {
            "title": data.get("title") or "",
            "uploader": data.get("author_name") or "",
            "description": "",
            "duration": None,
            "size_mb": None,
            "metadata_source": "youtube_oembed"
        }

    except Exception:
        return {}


def snippet_value(item, key, fallback=None):
    if isinstance(item, dict):
        return item.get(key, fallback)

    return getattr(item, key, fallback)


def get_youtube_transcript_fallback(video_id):
    if not video_id:
        return None

    try:
        transcript_obj = None

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            for lang in ["tr", "en"]:
                try:
                    transcript_obj = transcript_list.find_transcript([lang])
                    break
                except Exception:
                    pass

            if transcript_obj is None:
                try:
                    transcript_obj = transcript_list.find_generated_transcript(["tr", "en"])
                except Exception:
                    pass

            if transcript_obj is None:
                return None

            items = transcript_obj.fetch()
            language_code = getattr(transcript_obj, "language_code", "") or ""

        except Exception:
            try:
                api = YouTubeTranscriptApi()
                items = api.fetch(video_id, languages=["tr", "en"])
                language_code = ""
            except Exception:
                return None

        full_parts = []
        timed_parts = []
        first_3_parts = []

        for item in items:
            text = str(snippet_value(item, "text", "") or "").replace("\n", " ").strip()
            start = float(snippet_value(item, "start", 0) or 0)
            duration = float(snippet_value(item, "duration", 0) or 0)

            if not text:
                continue

            full_parts.append(text)

            timed_parts.append({
                "start": round(start, 2),
                "end": round(start + duration, 2),
                "text": text
            })

            if start <= 3:
                first_3_parts.append(text)

        full_transcript = " ".join(full_parts).strip()

        if len(full_transcript) > 60000:
            full_transcript = full_transcript[:60000] + "\n\n[Transkript uzun olduğu için kesildi.]"

        return {
            "language": language_code,
            "full_transcript": full_transcript,
            "timed_transcript": timed_parts[:200],
            "first_3_seconds_text": " ".join(first_3_parts).strip(),
            "strong_lines": extract_strong_lines(full_transcript),
            "weak_lines": "",
            "cta_text": detect_cta_text(full_transcript),
            "fallback_source": "youtube_transcript_api"
        }

    except Exception:
        return None


def insert_fallback_result(job, reason):
    url = job.get("url")
    platform = job.get("platform")
    link_type = job.get("link_type")
    video_id = job.get("video_id")

    metadata = {}

    if platform == "youtube":
        metadata = youtube_oembed_metadata(url)

    transcript_data = None

    if platform == "youtube":
        transcript_data = get_youtube_transcript_fallback(video_id)

    if transcript_data:
        insert_transcript(job, transcript_data)

    transcript = transcript_data.get("full_transcript", "") if transcript_data else ""
    first_3 = transcript_data.get("first_3_seconds_text", "") if transcript_data else ""
    cta_text = transcript_data.get("cta_text", "") if transcript_data else ""

    transcript_score = score_transcript(transcript)
    hook_score = score_hook(first_3)
    cta_score = 80 if cta_text else 35 if transcript else 0
    visual_score = 0

    avg_score = int((visual_score + transcript_score + hook_score + cta_score) / 4)

    if avg_score >= 65:
        viral_potential = "orta"
    elif avg_score >= 40:
        viral_potential = "düşük-orta"
    else:
        viral_potential = "düşük"

    result = {
        "job_id": job["id"],
        "video_url": url,
        "platform": platform,
        "link_type": link_type,
        "title": metadata.get("title") or "Video indirilemedi",
        "caption": metadata.get("description") or "",
        "duration_seconds": None,
        "uploader": metadata.get("uploader") or job.get("username") or "",
        "transcript_status": "completed_fallback" if transcript else "not_available",
        "visual_status": "download_failed_no_visual",
        "content_pillar": get_content_pillar(platform, transcript),
        "hook_type": "Fallback transkript hook analizi" if first_3 else "Video indirilemedi",
        "first_3_seconds_analysis": first_3 or "Video indirilemediği için ilk 3 saniye analizi yapılamadı.",
        "main_message": transcript[:1000] if transcript else "Video indirilemedi ve transcript bulunamadı.",
        "cta_analysis": cta_text or "CTA tespit edilemedi.",
        "edit_style": "Video dosyası indirilemediği için frame/edit analizi yapılamadı.",
        "visual_score": visual_score,
        "transcript_score": transcript_score,
        "hook_score": hook_score,
        "cta_score": cta_score,
        "viral_potential": viral_potential,
        "strengths": "Worker hata vermeden fallback analiz moduna geçti.",
        "weaknesses": "Video dosyası indirilemediği için görsel analiz yapılamadı.",
        "improved_hook": create_improved_hook(first_3, platform),
        "improved_caption": create_improved_caption(platform),
        "recommendations": "\n".join([
            "Bu link için video indirme başarısız oldu.",
            "YouTube transcript varsa metin analizi yapıldı.",
            "Görsel/frame analizi için farklı link veya farklı platform denenebilir.",
            "Bazı platformlar bot/runner erişimini engelleyebilir."
        ]),
        "raw_json": {
            "mode": "github_actions_fallback_analysis",
            "download_failed_reason": str(reason),
            "metadata": metadata,
            "has_transcript": bool(transcript),
            "temp_files_deleted": True
        },
        "created_at": now_iso()
    }

    insert_result(result)

    update_job(job["id"], {
        "status": "completed",
        "finished_at": now_iso(),
        "error": "Video indirilemedi, fallback analiz tamamlandı: " + str(reason)[:500]
    })

    log_event(
        job["id"],
        "warning",
        "fallback_completed",
        "Video indirilemedi ama fallback analiz tamamlandı.",
        str(reason)
    )

    return {
        "ok": True,
        "fallback": True,
        "job_id": job["id"],
        "has_transcript": bool(transcript),
        "title": metadata.get("title") or "",
        "reason": str(reason)
    }


def process_real_job(job):
    settings = get_settings()

    max_frames = safe_int(settings.get("MAX_FRAMES_PER_VIDEO"), 12)
    auto_delete = str(settings.get("AUTO_DELETE_TEMP", "TRUE")).upper() == "TRUE"

    job_id = job["id"]
    work_dir = BASE_TMP_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    download_meta = {}
    transcript_data = None
    scenes = []

    try:
        update_job(job_id, {
            "status": "processing",
            "started_at": now_iso(),
            "error": None
        })

        log_event(job_id, "info", "github_worker_start", "GitHub Actions worker başladı.", "")

        download_meta = download_video(job, work_dir, settings)
        video_path = download_meta["video_path"]

        if job.get("include_transcript", True):
            audio_path = work_dir / "audio.wav"
            extract_audio(video_path, audio_path)
            transcript_data = transcribe_audio(audio_path)
            insert_transcript(job, transcript_data)

        if job.get("include_visual", True):
            scenes = extract_frames(video_path, work_dir, max_frames)
            insert_scenes(job, scenes)

        result = build_result(job, download_meta, transcript_data, scenes)
        insert_result(result)

        update_job(job_id, {
            "status": "completed",
            "finished_at": now_iso(),
            "error": None
        })

        log_event(job_id, "info", "github_worker_completed", "GitHub Actions worker tamamlandı.", "")

        return {
            "ok": True,
            "fallback": False,
            "job_id": job_id,
            "title": download_meta.get("title"),
            "duration": download_meta.get("duration"),
            "size_mb": download_meta.get("size_mb"),
            "transcript": bool(transcript_data),
            "scene_count": len(scenes)
        }

    except Exception as e:
        reason = str(e)

        try:
            fallback = insert_fallback_result(job, reason)

            return {
                "ok": True,
                "fallback": True,
                "job_id": job_id,
                "title": fallback.get("title"),
                "duration": None,
                "size_mb": None,
                "transcript": fallback.get("has_transcript"),
                "scene_count": 0,
                "fallback_reason": reason
            }

        except Exception as fallback_error:
            update_job(job_id, {
                "status": "failed",
                "finished_at": now_iso(),
                "error": "Ana hata: " + reason + " | Fallback hata: " + str(fallback_error)
            })

            log_event(
                job_id,
                "error",
                "github_worker_failed",
                "GitHub worker ve fallback başarısız.",
                "Ana hata: " + reason + " | Fallback hata: " + str(fallback_error)
            )

            return {
                "ok": False,
                "job_id": job_id,
                "error": "Ana hata: " + reason + " | Fallback hata: " + str(fallback_error)
            }

    finally:
        if auto_delete:
            shutil.rmtree(work_dir, ignore_errors=True)
            log_event(job_id, "info", "cleanup", "Geçici dosyalar silindi.", str(work_dir))


def main():
    print("GitHub Actions worker başladı.")

    job = get_next_queued_job()

    if not job:
        print("Kuyrukta queued job yok.")
        return

    print("İşlenecek job:", job["id"])
    print("Platform:", job.get("platform"))
    print("Link tipi:", job.get("link_type"))
    print("URL:", job.get("url"))

    result = process_real_job(job)

    print("Worker sonucu:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
