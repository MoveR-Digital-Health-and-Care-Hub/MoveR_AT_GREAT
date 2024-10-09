import os
import torch

def list_files_recursively(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            current_path = os.path.join(dirpath, filename)
            if temp_block in current_path:  # day2_block1_position_5
                if current_path.endswith('_y.pt'):
                    print('\n\n----------------------------------------------')
                    print(current_path)
                    data = torch.load(current_path)
                    print(data.shape, 'trial length:', len(torch.unique(data[:,0])),  'pos', torch.unique(data[:,2]))
                    data[:,2] = 10
                    print(data.shape, 'trial length:', len(torch.unique(data[:,0])),  'pos', torch.unique(data[:,2]))
                    torch.save(data, current_path)

if __name__ == "__main__":

    p1 = '/home/kasia/AT_Great/AT_Great/data/processed_data_16x9feats' 
    p2 = '/home/kasia/AT_Great/AT_Great/data/processed_data_144x256feats_augstride10and10'
    root_directory = [p1, p2]
    for p in root_directory:
        print('----------------------- Participant:', p, '-----------------------')

        # block number either 1 or 2
        for b in [1,2]:
            temp_block = 'day2_block' +str(b)+'_position_10'
            list_files_recursively(p)
    