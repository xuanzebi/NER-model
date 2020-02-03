# BIO2BIEOS
def to_bioes(original_tags):
    def _change_prefix(original_tag, new_prefix):
        assert original_tag.find("-") > 0 and len(new_prefix) == 1
        chars = list(original_tag)
        chars[0] = new_prefix
        return "".join(chars)

    def _pop_replace_append(stack, bioes_sequence, new_prefix):
        tag = stack.pop()
        new_tag = _change_prefix(tag, new_prefix)
        bioes_sequence.append(new_tag)

    def _process_stack(stack, bioes_sequence):
        if len(stack) == 1:
            _pop_replace_append(stack, bioes_sequence, "S")
            # _pop_replace_append(stack, bioes_sequence, "U")
        else:
            recoded_stack = []
            _pop_replace_append(stack, recoded_stack, "E")
            # _pop_replace_append(stack, recoded_stack, "L")
            while len(stack) >= 2:
                _pop_replace_append(stack, recoded_stack, "I")
            _pop_replace_append(stack, recoded_stack, "B")
            recoded_stack.reverse()
            bioes_sequence.extend(recoded_stack)

    bioes_sequence = []
    stack = []

    for tag in original_tags:
        if tag == "O":
            if len(stack) == 0:
                bioes_sequence.append(tag)
            else:
                _process_stack(stack, bioes_sequence)
                bioes_sequence.append(tag)
        elif tag[0] == "I":
            if len(stack) == 0:
                stack.append(tag)
            else:
                this_type = tag[2:]
                prev_type = stack[-1][2:]
                if this_type == prev_type:
                    stack.append(tag)
                else:
                    _process_stack(stack, bioes_sequence)
                    stack.append(tag)
        elif tag[0] == "B":
            if len(stack) > 0:
                _process_stack(stack, bioes_sequence)
            stack.append(tag)
        else:
            raise ValueError("Invalid tag:", tag)

    if len(stack) > 0:
        _process_stack(stack, bioes_sequence)

    return bioes_sequence


# BIOES2BIO
def BIOES2BIO(input_file, output_file):
    print("Convert BIOES -> BIO for file:", input_file)
    with open(input_file, 'r') as in_file:
        fins = in_file.readlines()
    fout = open(output_file, 'w')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx] + " " + labels[idx] + "\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "E-" in labels[idx]:
                        fout.write(words[idx] + " I-" + label_type + "\n")
                    elif "S-" in labels[idx]:
                        fout.write(words[idx] + " B-" + label_type + "\n")
                    else:
                        fout.write(words[idx] + " " + labels[idx] + "\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1].upper())
    fout.close()
    print("BIO file generated:", output_file)


