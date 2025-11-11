with open('popular-names.txt','r') as f:
    lines = f.readlines()
    line_size = len(lines)
    split_line_size = line_size//10 +(1 if line_size % 10 else 0)
    for i in range(10):
        start = i*split_line_size
        end = (i+1)*split_line_size -1
        output = lines[start:end]
        if not output:
            break
        output_file = f'output_file{i+1}.txt'
        with open(output_file,'w') as out:
            out.writelines(output)
        