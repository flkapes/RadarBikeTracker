from video_processor import VideoProcessor
import tesseracts


def main():
    input_video_path = "GRMN0128.mp4"
    radar_data_path = "radar_data.json"
    output_dir = "image_output"
    fudge_factor = tesseracts.get_start_time(input_video_path)[1]

    video_processor = VideoProcessor(
        input_video_path=input_video_path,
        radar_data_path=radar_data_path,
        output_dir=output_dir,
        fudge_factor=fudge_factor,
    )
    video_processor.process_video()


if __name__ == "__main__":
    main()
