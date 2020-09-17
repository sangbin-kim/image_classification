import pandas as pd
import os
import shutil
import functions

# 1. Directory Setting
save_directory = r'D:/#.Secure Work Folder/1. Data/1. CMI/1. FLD/test4_result/5-4/'
image_directory = r"D:\#.Secure Work Folder\1. Data\1. CMI\1. FLD\test3\5-4\20200830"

# 2. Create DataFrame for results
result_df = pd.DataFrame(columns= ['ID', 'FILE', '1', '2', '3', 'JUDGE' ])

# 3. load trained model
model_save_path = r"C:/Users/LG/Desktop/ksb/3. CODE/model/"
filename = 'sepa_image_classifier_crop.pth'
model = functions.load_checkpoint(model_save_path, filename)

# 4. Classfiying images and reorganize image based on the JUDGE from model

for root, dirs, files in os.walk(image_directory):

    for file in files:

        if os.path.isdir(save_directory + root[-13:]) == False:
            os.mkdir(save_directory + root[-13:])
        if file.endswith("NG(0).JPG"):
            print(root[-13:])
            print(file)
            probs, classes = functions.predict(os.path.join(root, file), model)
            result_df = result_df.append({'ID': root[-13:],
                                          'FILE': file,
                                          '{}'.format(classes[0]): probs[0],
                                          '{}'.format(classes[1]): probs[1],
                                          '{}'.format(classes[2]): probs[2],
                                          'JUDGE': classes[0]}, ignore_index=True)
            judge = 'JUDGE_{}_'.format(classes[0]) + str(round(int(probs[0] * 100), 0)) + "_" + file
            print(judge)
            shutil.copy(os.path.join(root, file), save_directory + '/' + root[-13:] + '/' + judge)




# 5. Save results and judges
result_df.to_csv(save_directory+ 'result30.csv', index= False)

# 6. Judge based on folder
fcell_judge = pd.crosstab(result_df['ID'], result_df['JUDGE'])
folders = fcell_judge.index
cases = fcell_judge.columns
fcell_judge['FCell_Judge'] = None
for folder in folders:
    largest_value = -1
    largest_case = -1
    for case in cases:
        if fcell_judge.loc[folder, case] > largest_value:
            largest_value = fcell_judge.loc[folder, case]
            largest_case = str(case)

        elif fcell_judge.loc[folder, case] == largest_value:
            largest_case += ' or ' + str(case)

    fcell_judge.loc[folder, 'FCell_Judge'] = largest_case

fcell_judge.to_csv(save_directory + 'folded_cell_judge30.csv')