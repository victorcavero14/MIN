import sys

def num_instances(s):
    # type: (str) -> int
    # s = "1x"
    # return "1"
    return int(s[:-1])

# def case_txt_2_csv():


def convert(txt_filename, csv_filename):
    # type: (str, str) -> None
    f_in = open(txt_filename, "r")
    f_out = open(csv_filename, "w")

    # l_in = "1x Case1   A B C D"
    # l_in.split()
    # ['1x', 'Case1', 'A', 'B', 'C', 'D']

    f_out.write("case,event\n")
    for l_in in f_in:
        case = l_in.split()
        n = num_instances(case[0])
        case_name = case[1]
        events = case[2:]

        for i in range(n):
            for e in events:
                l_out = "{0},{1}\n".format(case_name, e)
                f_out.write(l_out)

    f_in.close()
    f_out.close()


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])