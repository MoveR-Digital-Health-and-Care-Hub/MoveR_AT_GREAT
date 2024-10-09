import os


def list_files_recursively(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            current_path = os.path.join(dirpath, filename)
            if temp_block in current_path:  # day2_block1_position_5
                new_path = current_path.replace(temp_block, 'day2_block'+str(b)+'_position_10')
                os.rename(current_path, new_path)  # Rename the file
                print(f"Renamed: {current_path} to {new_path}")




if __name__ == "__main__":

    p1 = '/home/kasia/AT_Great/AT_Great/data/processed_data_16x9feats' 
    p2 = '/home/kasia/AT_Great/AT_Great/data/processed_data_144x256feats_augstride10and10'
    # root_directory = r'/home/kasia/AT_Great/AT_Great/data/processed_data_16x9feats'
    #  # r'/home/kasia/AT_Great/AT_Great/data/processed_data_144x256feats_augstride10and10'

    root_directory = [p1, p2]
    for p in root_directory:
        print('----------------------- Participant:', p, '-----------------------')


        blocks=[1,2] # 1 # block number either 1 or 2
        for b in blocks:
            temp_block = 'day2_block' +str(b)+'_position_5'
            list_files_recursively(p)