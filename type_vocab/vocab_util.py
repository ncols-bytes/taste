import random
import json

def load_type_vocab(file_path):
    type_vocab = {}

    with open(file_path, "r") as f:
        for line in f:
            index, t = line.strip().split('\t')
            type_vocab[t] = int(index)
    return type_vocab


def random_select_types(all_types, select_n, random_seed=0):
    random.seed(0)
    random.shuffle(all_types)
    return all_types[:select_n]


if __name__ == "__main__":
    type_vocab = load_type_vocab('type_vocab/wikitable/type_vocab_255.txt')
    all_types = list(type_vocab.keys())

    n_s = [240,190,180,130,100,50]
    for n in n_s:
        all_types.sort()
        seleced_types = random_select_types(all_types, n)
        with open(f'type_vocab/wikitable/type_vocab_{n}.txt', 'w') as f:
            for i, seleced_type in enumerate(seleced_types):
                f.write(str(i) + '\t' + seleced_type + '\n')
        
        no_type_cnt = 0
        col_cnt = 0
        with open('data/wikitable/test.table_col_type.json', 'r') as f:
            data = json.load(f)
            for table_idx in range(len(data)):
                headers = data[table_idx][5]
                types = data[table_idx][7]

                for column_idx in range(len(types)):
                    col_cnt += 1
                    has_type = False
                    for type in types[column_idx]:
                        if type in seleced_types:
                            has_type = True
                            break
                    if not has_type:
                        no_type_cnt += 1

        print(len(seleced_types), col_cnt, no_type_cnt, no_type_cnt / col_cnt)