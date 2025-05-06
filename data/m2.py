import re
import random


class MotifSampler:
    def __init__(self, input_str):
        self.input_str = input_str
        self.letter_segments = []
        self.number_segments = []
        self.sampled_lengths = []

        self.parse_input()
        self.sample_lengths()

    def parse_input(self):
        segments = self.input_str.split(',')
        self.letter_segments = [seg for seg in segments if re.match(r'^[A-Za-z]', seg)]
        self.number_segments = [seg for seg in segments if not re.match(r'^[A-Za-z]', seg)]

    def sample_lengths(self):
        self.sampled_lengths = []
        for seg in self.number_segments:
            start, end = map(int, seg.split('-'))
            self.sampled_lengths.append(random.randint(start, end))

    def get_results(self):
        return {
            "letter_segments": self.letter_segments,
            "number_segments": self.number_segments,
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


class MotifSamplerMultiChain:
    def __init__(self, input_str):
        self.input_str = input_str
        self.chain_segments = []
        self.sampled_segments = []

        self.parse_input()
        self.sample_lengths()

    def parse_input(self):
        self.chain_segments = self.input_str.split('/')

    def sample_lengths(self):
        self.sampled_segments = []
        for segment in self.chain_segments:
            sampled_segment = []
            for part in segment.split(','):
                if '-' in part and re.match(r'^\d+-\d+$', part):
                    start, end = map(int, part.split('-'))
                    sampled_segment.append(str(random.randint(start, end)))
                else:
                    sampled_segment.append(part)
            self.sampled_segments.append(','.join(sampled_segment))

    def get_final_output(self):
        return '/'.join(self.sampled_segments)

    def get_results(self):
        return {
            "chain_segments": self.chain_segments,
            "sampled_lengths": self.sample_lengths
        }



if __name__ == '__main__':
    # 使用示例
    input_str = "E400-510/20-45,A24-42,4-10,A64-82,0-5"

    sampler = MotifSamplerMultiChain(input_str)
    results = sampler.get_results()

    # print(f"Letter segments: {results['letter_segments']}")
    # print(f"Number segments: {results['number_segments']}")
    # print(f"Sampled lengths: {results['sampled_lengths']}")

    final_output = sampler.get_final_output()
    print(f"Final output: {final_output}")
