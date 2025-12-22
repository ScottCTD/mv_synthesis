from scenedetect import ContentDetector, detect, split_video_ffmpeg


def main():
    paths = []
    for i, path in enumerate(paths):
        scenes = detect(path, ContentDetector())
        split_video_ffmpeg(
            path, scenes, output_dir=f"./output_scenes/{i + 3}/")


if __name__ == "__main__":
    main()
