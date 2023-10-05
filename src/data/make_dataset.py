# -*- coding: utf-8 -*-
# import click
import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv
from pydub import AudioSegment

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path()


# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


def cut_audio(input_audio_path, segment_length, output_folder):
    # Load the input audio file
    audio = AudioSegment.from_file(input_audio_path)

    # Calculate the segment length in milliseconds
    segment_length_ms = segment_length * 1000  # Convert seconds to milliseconds

    # Initialize the start time for the first segment
    start_time = 0
    segment_index = 1

    while start_time < len(audio):
        # Calculate the end time for the current segment
        end_time = start_time + segment_length_ms

        # Extract the segment
        segment = audio[start_time:end_time]

        # Define the output file name (e.g., segment_1.wav, segment_2.wav, etc.)
        output_filename = f"segment_{segment_index}.wav"

        # Save the segment to the specified output folder
        output_path = f"{output_folder}/{output_filename}"
        segment.export(output_path, format="wav")

        # Update the start time and segment index for the next segment
        start_time = end_time
        segment_index += 1

if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    # main()
    # Replace with the path to your input audio file
    input_audio_path = "data/raw/speaker_Allan_Adams.mp3"
    segment_lengths = 15 # Specify the segment lengths in seconds
    # Specify the folder where segments will be saved
    output_folder = "data/processed/Allan_Adams"

    cut_audio(input_audio_path, segment_lengths, output_folder)
