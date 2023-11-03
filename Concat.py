import os 

directory_path_sub = '/scratch/nia/minjay/231026_obj_sub'
directory_path = '/scratch/nia/minjay/231026_obj'

result_path = '/scratch/nia/minjay/231026_obj'

# List all files and directories in the specified directory
ori = os.listdir(directory_path)
sub = os.listdir(directory_path_sub)

for sub_dir in sub :

    if sub_dir == '.DS_Store' :
            continue

    sub_dir_file = os.listdir( os.path.join(directory_path_sub, sub_dir) )

    for file in sub_dir_file :

        subfile = os.path.join(directory_path_sub, sub_dir,file)
        orifile = os.path.join(directory_path, sub_dir[:-4],file)
        savefile = os.path.join(result_path, sub_dir[:-4],file)
        savedir = os.path.join(result_path, sub_dir[:-4])

        print(subfile)
        
        try :
            os.makedirs(savedir)
        except :
            print(savedir)

        # Open the input files for reading
        with open(subfile, 'r', encoding='euc-kr') as sub, open(orifile, 'r', encoding='euc-kr') as ori:
            lines1 = ori.readlines()
            lines2 = sub.readlines()

        # Create a list to store the concatenated data
        concatenated_data = []
        
        first = 0
        # Iterate through lines from both files and concatenate the second column
        for line1, line2 in zip(lines1, lines2):

            columns1 = line1.strip().split()
            columns2 = line2.strip().split()

            str_gen = ''
            
            for idx, col in enumerate(columns1) :
                if idx == 0 :
                    if first == 0 :
                        first = 1
                        str_gen += str(float(col) + float(columns2[0]))
                    else :
                        str_gen += col
                else :
                    str_gen += ' ' + col 
            
            for idx, col in enumerate(columns2) :
                if idx == 0 :
                    continue
                str_gen += ' ' + col 

            concatenated_data.append(str_gen)



        # Write the concatenated data to the output file
        with open(savefile, 'w') as output_file:
            output_file.write('\n'.join(concatenated_data))