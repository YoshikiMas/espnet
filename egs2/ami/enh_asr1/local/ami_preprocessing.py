import argparse
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import logging
import os
from pathlib import Path


@dataclass
class Utterance:
    meeting_id: str
    speaker_id: str
    start_time: float
    end_time: float
    text: str


@dataclass
class TwoSpeakerSegments:
    segment_id: str
    meeting_id: str
    speaker_id: str
    start_time: float
    end_time: float
    speaker1: Utterance
    speaker2: Utterance


def load_meeting(file_path: str, mic: str = "sdm1"):
    meetings = {}
    current_meeting_id = None
    current_meeting = []

    dset = Path(file_path).stem
    wav_scp_path = Path(file_path).parent.parent.joinpath(mic).joinpath(dset).joinpath("wav.scp")
    appropriate_meeting_ids = set()
    with open(wav_scp_path, "r") as file:
        for line in file:
            line = line.strip()
            meeting_id, _ = line.split(maxsplit=1)
            appropriate_meeting_ids.add(meeting_id.split("_")[1])

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            meeting_id = parts[0]
            speaker_id = parts[2]
            start_time = float(parts[3])
            end_time = float(parts[4])
            text = " ".join(parts[5:])

            if meeting_id not in appropriate_meeting_ids:
                continue

            current_utterance = Utterance(
                meeting_id,
                speaker_id,
                start_time,
                end_time,
                text,
            )

            # NOTE (Yoshiki): Store the meeting if meeting_id of the current utterance is different from the previous one.
            if current_utterance.meeting_id != current_meeting_id:
                if current_meeting_id is not None:
                    current_meeting.sort(key=lambda x: x.start_time)
                    meetings[current_meeting_id] = deque(current_meeting)

                current_meeting = []

            current_meeting.append(current_utterance)
            current_meeting_id = current_utterance.meeting_id

        # NOTE (Yoshiki): Store the last meeting.
        current_meeting.sort(key=lambda x: x.start_time)
        meetings[current_meeting_id] = deque(current_meeting)

    return meetings


def construct_two_speaker_segment(current_utterances):
    if len(current_utterances) == 2:
        meeting_id = current_utterances[0].meeting_id
        speaker_id = current_utterances[0].speaker_id + "_" + current_utterances[1].speaker_id
        start_time = current_utterances[0].start_time
        end_time = max(current_utterances[0].end_time, current_utterances[1].end_time)
        time_info = f"{int(100 * start_time + 0.5):07d}_{int(100 * end_time + 0.5):07d}"
        segment_id = f"AMI_{speaker_id}_{meeting_id}_{time_info}"

        two_speaker_segment = TwoSpeakerSegments(
            segment_id,
            meeting_id,
            speaker_id,
            start_time,
            end_time,
            current_utterances[0],
            current_utterances[1],
        )
        return two_speaker_segment

    else:
        # NOTE (Yoshiki): The last successive utterances are overlapped.
        speakers = set()
        for x in current_utterances:
            speakers.add(x.speaker_id)

        if len(speakers) != 2:
            return

        else:
            # NOTE (Yoshiki): The last overlapped segements are uttered by two speakers.
            end_time = 0.0
            speaker1_utterance = current_utterances[0]
            speaker2_utterance = None
            for x in current_utterances[1:]:
                end_time = max(end_time, x.end_time)
                if x.speaker_id == speaker1_utterance.speaker_id:
                    assert (
                        speaker1_utterance.end_time <= x.end_time
                    ), "The utterances of the same speaker should not overlpa"
                    speaker1_utterance.end_time = x.end_time
                    speaker1_utterance.text += " " + x.text
                else:
                    if speaker2_utterance is None:
                        speaker2_utterance = x
                    else:
                        assert (
                            speaker2_utterance.end_time <= x.end_time
                        ), "The utterances of the same speaker should not overlpa"
                        speaker2_utterance.end_time = max(speaker2_utterance.end_time, x.end_time)
                        speaker2_utterance.text += " " + x.text

            meeting_id = speaker1_utterance.meeting_id
            speaker_id = speaker1_utterance.speaker_id + "_" + speaker2_utterance.speaker_id
            start_time = speaker1_utterance.start_time
            time_info = f"{int(100 * start_time + 0.5):07d}_{int(100 * end_time + 0.5):07d}"
            segment_id = f"AMI_{speaker_id}_{meeting_id}_{time_info}"

            two_speaker_segment = TwoSpeakerSegments(
                segment_id,
                meeting_id,
                speaker_id,
                start_time,
                end_time,
                speaker1_utterance,
                speaker2_utterance,
            )

            return two_speaker_segment


def detect_two_speaker_overlpas(meeting: deque, vocal_sounds: set = set()):
    current_utterances = [meeting.popleft()]
    two_speaker_segments = []
    while len(meeting):
        utterance = meeting.popleft()
        if utterance.text in vocal_sounds:
            # NOTE (Yoshiki): This is for additional filtering
            continue

        if current_utterances[-1].end_time <= utterance.start_time:
            if len(current_utterances) >= 2:
                # NOTE (Yoshiki): The last successive utterances are overlapped and stored in a list.
                two_speaker_segment = construct_two_speaker_segment(current_utterances)
                if two_speaker_segment is not None:
                    two_speaker_segments.append(two_speaker_segment)

            current_utterances = [utterance]

        else:
            # NOTE (Yoshiki): The new utterance is overalpped with the previous utterenace
            current_utterances.append(utterance)

    # NOTE (Yoshiki): This part handes the last succesive utterances.
    if len(current_utterances) >= 2:
        two_speaker_segment = construct_two_speaker_segment(current_utterances)
        if two_speaker_segment is not None:
            two_speaker_segments.append(two_speaker_segment)

    return two_speaker_segments


def save_two_speaker_segments(output_dir: str, ami_dir: str, two_speaker_segments: list):
    # NOTE (Yoshiki): spk1.scp and spk2.scp are dummy. It might be better to use IHM recordings as pseudo targets.
    # NOTE (Yoshiki): spk2utt should be preoared by utils/
    sox = "sox -c 1 -t wavpcm -e signed-integer"

    output_path = Path(output_dir)
    with open(output_path.joinpath("wav.scp"), "w") as wav_scp_f, open(
        output_path.joinpath("spk1.scp"), "w"
    ) as spk1_scp_f, open(output_path.joinpath("spk2.scp"), "w") as spk2_scp_f, open(
        output_path.joinpath("text_spk1"), "w"
    ) as text_spk1_f, open(
        output_path.joinpath("text_spk2"), "w"
    ) as text_spk2_f, open(
        output_path.joinpath("utt2spk"), "w"
    ) as utt2spk_f, open(
        output_path.joinpath("segments"), "w"
    ) as segment:
        wav_scp_f.truncate()
        spk1_scp_f.truncate()
        spk2_scp_f.truncate()
        text_spk1_f.truncate()
        text_spk2_f.truncate()
        utt2spk_f.truncate()
        segment.truncate()

        for two_speaker_segment in two_speaker_segments:
            segment_id = two_speaker_segment.segment_id
            meeting_id = two_speaker_segment.meeting_id
            audio = f"{sox} {ami_dir}/{meeting_id}/audio/{meeting_id}.Array1-01.wav -t wavpcm - |"
            start_time = str(two_speaker_segment.start_time)
            end_time = str(two_speaker_segment.end_time)

            wav_scp_f.write(segment_id + " " + audio + "\n")
            spk1_scp_f.write(segment_id + " " + audio + "\n")
            spk2_scp_f.write(segment_id + " " + audio + "\n")
            text_spk1_f.write(segment_id + " " + two_speaker_segment.speaker1.text + "\n")
            text_spk2_f.write(segment_id + " " + two_speaker_segment.speaker2.text + "\n")
            utt2spk_f.write(segment_id + " " + two_speaker_segment.speaker_id + "\n")
            segment.write(segment_id + " " + segment_id + " " + start_time + " " + end_time + "\n")


def main():
    """
    EXAMPLE:
        python ami_preprocessing.py \
            /mm0/masuyama/AudioRep/espnet/egs2/ami/asr1_ihm/data/local/annotations \
            /mm0/masuyama/local_datasets/original/AMI \
            ./
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("ami_espnet_annotations", type=str)
    parser.add_argument("ami", type=str)
    parser.add_argument("data", type=str)
    args = parser.parse_args()

    data_path = Path(args.data)
    espnet_annotations_path = Path(args.ami_espnet_annotations)

    for dset in ["train", "dev", "eval"]:
        output_dir = data_path.joinpath(dset)
        os.makedirs(output_dir, exist_ok=True)
        file_path = espnet_annotations_path.joinpath(dset + ".txt")
        meetings = load_meeting(file_path)

        two_speaker_segments = []
        for meeting in meetings.values():
            two_speaker_segments += detect_two_speaker_overlpas(deepcopy(meeting))

        two_speaker_segments.sort(key=lambda x: x.segment_id)
        save_two_speaker_segments(output_dir, args.ami, two_speaker_segments)


if __name__ == "__main__":
    main()
