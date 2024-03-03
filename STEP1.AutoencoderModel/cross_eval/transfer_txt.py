import os

val_img=[]
val_lbl=[]
val_name=[]
val_pseudo_lbl=[]
with open('recon/AbdomenAtlas_8k_raw.txt', 'a+') as f:
    for line in open(os.path.join('recon/AbdomenAtlas_8k.txt')):
        name = line.strip().split()[1].split('.')[0]
        img_path = line.strip().split()[0]
        if '01_Multi-Atlas_Labeling' in img_path:
            case_name = img_path.split('01_Multi-Atlas_Labeling_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('01_Multi-Atlas_Labeling', 'img', case_name+'.nii.gz')
            label_name = 'label'+case_name.split('img')[1]
            save_label = os.path.join('01_Multi-Atlas_Labeling', 'label', label_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '02_TCIA_Pancreas-CT' in img_path:
            case_name = img_path.split('02_TCIA_Pancreas-CT_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('02_TCIA_Pancreas-CT', 'img', case_name+'.nii.gz')
            save_label = os.path.join('02_TCIA_Pancreas-CT', 'pancreas_label', case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '03_CHAOS' in img_path:
            case_name = img_path.split('03_CHAOS_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('03_CHAOS/ct', 'img', case_name+'.nii.gz')
            label_name = case_name.split('_image')[0]
            save_label = os.path.join('03_CHAOS/ct', 'liver_label', label_name+'_segmentation.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '05_KiTS' in img_path:
            case_name = img_path.split('05_KiTS_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('05_KiTS', 'img', case_name+'.nii.gz')
            label_name = 'label'+case_name.split('img')[1]
            save_label = os.path.join('05_KiTS', 'label', label_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '07_WORD' in img_path:
            case_name = img_path.split('07_WORD_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('07_WORD', 'img', case_name+'.nii.gz')
            save_label = os.path.join('07_WORD', 'label', case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '08_AbdomenCT-1K' in img_path:
            case_name = img_path.split('08_AbdomenCT-1K_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('08_AbdomenCT-1K', 'img', case_name+'.nii.gz')
            label_name = case_name.split('_0000')[0]
            save_label = os.path.join('08_AbdomenCT-1K', 'label', label_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '09_AMOS' in img_path:
            case_name = img_path.split('09_AMOS_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('09_AMOS', 'img', case_name+'.nii.gz')
            save_label = os.path.join('09_AMOS', 'label', case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '10_Decathlon_colon' in img_path:
            case_name = img_path.split('10_Decathlon_colon_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('10_Decathlon/Task10_Colon', 'imagesTr', 'colon_'+case_name+'.nii.gz')
            save_label = os.path.join('10_Decathlon/Task10_Colon', 'labelsTr', 'colon_'+case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '10_Decathlon_hepaticvessel' in img_path:
            case_name = img_path.split('10_Decathlon_hepaticvessel_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('10_Decathlon/Task08_HepaticVessel', 'imagesTr', 'hepaticvessel_'+case_name+'.nii.gz')
            save_label = os.path.join('10_Decathlon/Task08_HepaticVessel', 'labelsTr', 'hepaticvessel_'+case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '10_Decathlon_liver' in img_path:
            case_name = img_path.split('10_Decathlon_liver_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('10_Decathlon/Task03_Liver', 'imagesTr', 'liver_'+case_name+'.nii.gz')
            save_label = os.path.join('10_Decathlon/Task03_Liver', 'labelsTr', 'liver_'+case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '10_Decathlon_lung' in img_path:
            case_name = img_path.split('10_Decathlon_lung_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('10_Decathlon/Task06_Lung', 'imagesTr', 'lung_'+case_name+'.nii.gz')
            save_label = os.path.join('10_Decathlon/Task06_Lung', 'labelsTr', 'lung_'+case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '10_Decathlon_pancreas' in img_path:
            case_name = img_path.split('10_Decathlon_pancreas_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('10_Decathlon/Task07_Pancreas', 'imagesTr', 'pancreas_'+case_name+'.nii.gz')
            save_label = os.path.join('10_Decathlon/Task07_Pancreas', 'labelsTr', 'pancreas_'+case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '10_Decathlon_spleen' in img_path:
            case_name = img_path.split('10_Decathlon_spleen_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('10_Decathlon/Task09_Spleen', 'imagesTr', 'spleen_'+case_name+'.nii.gz')
            save_label = os.path.join('10_Decathlon/Task09_Spleen', 'labelsTr', 'spleen_'+case_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '12_CT-ORG' in img_path:
            case_name = img_path.split('12_CT-ORG_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('12_CT-ORG', 'img', case_name+'.nii.gz')
            label_name = case_name.split('volume')[1]
            save_label = os.path.join('12_CT-ORG', 'label', 'label'+label_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '13_AbdomenCT-12organ' in img_path:
            case_name = img_path.split('13_AbdomenCT-12organ_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('13_AbdomenCT-12organ', 'img', case_name+'.nii.gz')
            label_name = case_name.split('_0000')[0]
            save_label = os.path.join('13_AbdomenCT-12organ', 'label', label_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')
        if '18_FLARE23' in img_path:
            case_name = img_path.split('18_FLARE23_')[1]
            case_name = case_name.split('/ct.nii.gz')[0]
            save_img = os.path.join('18_FLARE23', 'imagesTr2200', case_name+'.nii.gz')
            label_name = case_name.split('_0000')[0]
            save_label = os.path.join('18_FLARE23', 'labelsTr2200', label_name+'.nii.gz')
            f.write('{}        {}'.format(save_img, save_label))
            f.write('\n')