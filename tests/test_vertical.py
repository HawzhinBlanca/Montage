import subprocess, json, pathlib
def test_vertical():
    out="tests/out.mp4"
    subprocess.run(["python","-m","src.run_pipeline",
                    "tests/data/lecture.mp4","--mode","smart",
                    "--out",out],check=True)
    meta=json.loads(subprocess.check_output(
        ["ffprobe","-v","quiet","-print_format","json","-show_streams",out]))
    v=next(s for s in meta["streams"] if s["codec_type"]=="video")
    assert v["width"]==1080 and v["height"]==1920