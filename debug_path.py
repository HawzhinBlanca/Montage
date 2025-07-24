import os

test_path = "/tmp/test_video.mp4"
abs_path = os.path.realpath(test_path)
print(f"Input: {test_path}")
print(f"Absolute: {abs_path}")

allowed_bases = [
    os.path.realpath(os.getcwd()),
    os.path.realpath(os.path.expanduser("~/Videos")),
    os.path.realpath(os.path.expanduser("~/Downloads")),
    os.path.realpath("/tmp"),
    os.path.realpath("/var/tmp"),
]

print("\nAllowed bases:")
for i, base in enumerate(allowed_bases):
    print(f"{i}: {base}")
    print(f"   Starts with? {abs_path.startswith(base)}")