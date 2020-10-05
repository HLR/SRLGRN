import json
import sys

def read_json(fpath):
    data = json.load(open(fpath, 'r'))
    print(f'loading data from:{fpath}')
    return data

def write_json(data, outfile):
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    ans_path = sys.argv[1]
    output_path = sys.argv[2]

    ans = read_json(ans_path)

    joint = {}
    joint['answer'] = ans
    joint['sp'] = {}

    write_json(joint, output_path)
    print('finish')

if __name__ == '__main__':
    main()



    


