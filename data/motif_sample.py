import re
import random


class MotifSampler:
    def __init__(self, input_str, total_length):
        self.input_str = input_str
        self.total_length = total_length
        self.letter_segments = []
        self.number_segments = []
        self.letter_length = 0
        self.random_sample_total_length = 0
        self.sampled_lengths = []

        self.parse_input()
        self.calculate_random_sample_total_length()
        self.sample_lengths()

    def parse_input(self):
        segments = self.input_str.split(',')
        self.letter_segments = [seg for seg in segments if re.match(r'^[A-Za-z]', seg)]
        self.number_segments = [seg for seg in segments if not re.match(r'^[A-Za-z]', seg)]

        self.letter_length = 0
        for seg in self.letter_segments:
            match = re.match(r'^[A-Za-z](\d+)-(\d+)', seg)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                self.letter_length += end - start + 1

    def calculate_random_sample_total_length(self):
        self.random_sample_total_length = self.total_length - self.letter_length

    def sample_lengths(self):
        number_ranges = [(int(seg.split('-')[0]), int(seg.split('-')[1])) for seg in self.number_segments]
        self.sampled_lengths = self.find_lengths(number_ranges, self.random_sample_total_length)
        if not self.sampled_lengths:
            raise ValueError("Unable to find valid sampled lengths within the given ranges and total length.")

    def find_lengths(self, ranges, total_length):
        if not ranges:
            return [] if total_length == 0 else None

        start, end = ranges[0]
        possible_lengths = list(range(start, end + 1))
        random.shuffle(possible_lengths)  # 打乱可能的长度顺序

        for length in possible_lengths:
            if length <= total_length:
                remaining_lengths = self.find_lengths(ranges[1:], total_length - length)
                if remaining_lengths is not None:
                    return [length] + remaining_lengths

        return None

    def get_results(self):
        return {
            "letter_segments": self.letter_segments,
            "number_segments": self.number_segments,
            "total_motif_length": self.letter_length,
            "random_sample_total_length": self.random_sample_total_length,
            "sampled_lengths": self.sampled_lengths
        }

    def get_final_output(self):
        combined_segments = []
        num_idx = 0
        letter_idx = 0

        for segment in self.input_str.split(','):
            if re.match(r'^[A-Za-z]', segment):
                combined_segments.append(self.letter_segments[letter_idx])
                letter_idx += 1
            else:
                combined_segments.append(str(self.sampled_lengths[num_idx]))
                num_idx += 1

        return ','.join(combined_segments)


    def get_individual_sample_output(self):
        combined_segments = []
        letter_idx = 0
        sampled_lengths = self.sample_individual_lengths()

        for segment in self.input_str.split(','):
            if re.match(r'^[A-Za-z]', segment):
                combined_segments.append(self.letter_segments[letter_idx])
                letter_idx += 1
            else:
                combined_segments.append(str(sampled_lengths.pop(0)))

        return ','.join(combined_segments)

if __name__ == '__main__':


    # 使用示例
    input_str = "0-35,A45-65,0-35"
    total_length = 56

    sampler = MotifSampler(input_str, total_length)
    results = sampler.get_results()

    print(f"Letter segments: {results['letter_segments']}")
    print(f"Number segments: {results['number_segments']}")
    print(f"Total motif length: {results['total_motif_length']}")
    print(f"Random sample total length: {results['random_sample_total_length']}")
    print(f"Sampled lengths: {results['sampled_lengths']}")

    final_output = sampler.get_final_output()
    print(f"Final output: {final_output}")

    individual_sample_output = sampler.get_individual_sample_output()
    print(f"Final output (individual sampling): {individual_sample_output}")
