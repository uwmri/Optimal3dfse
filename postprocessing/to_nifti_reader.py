# save as anonymized nifti & keys for reader study

import nibabel as nib
import h5py
import numpy as np
import os
import matplotlib
import glob
import csv
import random
import uuid

# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from nibabel.orientations import axcodes2ornt, ornt_transform, io_orientation, apply_orientation, aff2axcodes, inv_ornt_aff


def generate_patient_id(group_name):
    rng = random.Random(hash(group_name) & 0xffffffff)
    base_id = ''.join(rng.choices('ABCDEFGHJKLMNPQRSTUVWXYZ23456789', k=6))
    return base_id


# header_source = nib.load(r'D:\DATA\Scans_d\v12ctang_01914_2024-05-29\01914_00008_Sag_T2_FLAIR_CUBE\raw_data\from_dicom_Sag_T2_FLAIR_CUBE_20240529122121_8.nii')
# --- Direction cosines from DICOM tag
row_cosines = np.array([0, 1, 0])     # Rows along posterior
col_cosines = np.array([0, 0, -1])    # Columns along inferior
slice_cosines = np.cross(row_cosines, col_cosines)  # Slices toward right (cross-product)

# --- Image origin
origin = np.array([0,0,0])

voxel_size = (250/256, 250/256, 204.8/209)
# --- Build 4x4 affine matrix
affine = np.eye(4)
affine[:3, 0] = row_cosines * voxel_size[0]
affine[:3, 1] = col_cosines * voxel_size[1]
affine[:3, 2] = slice_cosines * voxel_size[2]
affine[:3, 3] = origin

recon_type='train'
# scan_dir = r'D:\SSD\Data\Scans_i\Opt3dfse_ADRC_add_on\AEA_08953_2024-10-15\08953_00012_Optimal_3DFLAIR_c\raw_data'

scan_roots = [r'S:\Opt3dfse_ADRC_add_on', r'D:\SSD\Data\Scans_i\Opt3dfse_ADRC_add_on']
scan_types = [
    '*Sag_T2_FLAIR_CUBE',
    '*Opt*_3DFLAIR_c',
    '*Opt*_3DFLAIR'
]
save_dir = r'S:\nifti'
log_file = os.path.join(scan_roots[0], 'generate_nifti.csv')
with open(log_file, mode='w') as log:
    log.write('Patient, ScanName, patientUID, RandID\n')

for scan_root in scan_roots:
    patients = [name for name in os.listdir(scan_root)
        if os.path.isdir(os.path.join(scan_root, name))]

    for fi in range(len(patients)):
        patient_path = os.path.join(scan_root, patients[fi])

        patient_uid  = uuid.uuid4().hex[:8]
        randIDs = np.random.permutation([0, 1])
        randIDs = np.insert(randIDs, 0, 2)

        for si, scan_type in enumerate(scan_types):
            scan_dir = os.path.join(glob.glob(os.path.join(patient_path, scan_type))[0], 'raw_data')

            if recon_type == 'orc':
                filename = 'KSPACE_001'
                hf = h5py.File(name=os.path.join(scan_dir, filename+'.h5'), mode='r')
                sos_act = hf['sos_act'][()]
                sos_act = sos_act.astype(np.float32)

                # rotate to what freesurfer expects
                sos_act = np.rot90(sos_act, k=2, axes=(1, 2))

                truth = nib.Nifti1Image(sos_act, np.eye(4))
                truth = nib.Nifti1Image(truth.get_fdata(), truth.affine, header=header_source.header)

                try:
                    os.remove(os.path.join(scan_dir, filename + '.nii'))
                except:
                    pass
                nib.save(truth, os.path.join(scan_dir, filename + '.nii'))


            else:
                recon_types = ['l1w_z216', 'train', 'train']
                recon_type = recon_types[si]

                # inference or real scan
                if recon_type == 'train':
                    filename = glob.glob(os.path.join(scan_dir, 'recon_denoiserTrue_phasefittingTrue_kwFalse*.h5'))[0]
                    hf = h5py.File(name=filename, mode='r')
                    mag = hf['train_mag_biasCorr'][()]
                elif recon_type == 'l1w_z216':
                    filename = glob.glob(os.path.join(scan_dir, 'recon_l1w_z216*.h5'))[0]
                    hf = h5py.File(name=filename, mode='r')
                    mag = hf['Images_bc'][()]
                print(f'exporting from {filename}')
                mag = nib.Nifti1Image(mag, np.eye(4))
                mag.set_sform(affine, code=1)
                mag.set_qform(np.eye(4), code=0)

                target_ornt = axcodes2ornt(('R', 'A', 'S'))
                transform = ornt_transform(io_orientation(mag.affine), target_ornt)
                mag_ax = apply_orientation(mag.get_fdata(), transform)
                new_affine = mag.affine @ inv_ornt_aff(transform, mag.shape)
                mag_ax_img = nib.Nifti1Image(mag_ax, new_affine)

                randID = randIDs[si]
                # mag_header = mag.header
                # mag_header['descrip'] = f'{patient_uid}_{randID}'
                print(f"{patients[fi]},{scan_type},{patient_uid},{randID}")

                filename_nifti = f'{patient_uid}_{randID}.nii.gz'
                try:
                    os.remove(os.path.join(save_dir, filename_nifti))
                except:
                    pass
                nib.save(mag_ax_img, os.path.join(save_dir, filename_nifti ))

                with open(log_file, 'a') as log:
                    log.write(f"{patients[fi]},{scan_type},{patient_uid},{randID}\n")

        with open(log_file, 'a') as log:
            log.write(f"\n")


# # make the scoring sheet
# import pandas as pd
# from openpyxl import Workbook
# from openpyxl.styles import PatternFill, Alignment
#
# df = pd.read_csv(os.path.join(scan_roots[0],'generate_nifti.csv'))
#
# df = df[df["RandID"] != 2]
# uids = df['patientUID'].unique()
#
# # Set up workbook and worksheet
# wb = Workbook()
# ws = wb.active
# ws.title = "Results"
#
# # Define fills
# gray_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")      # light gray
# blue_fill = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")      # light blue
# wrap_text = Alignment(wrap_text=True, vertical="top")
#
# # Starting row
# # Start writing rows
# row_idx = 1
# for uid in uids:
#     # Row 1: UID header
#     ws.cell(row=row_idx, column=1, value=uid)
#     ws.cell(row=row_idx, column=2, value="Ranking")
#     ws.cell(row=row_idx, column=3, value="Better overall image quality:")
#     ws.cell(row=row_idx, column=4, value="")
#     ws.cell(row=row_idx, column=4).fill = blue_fill
#     ws.cell(row=row_idx, column=4).alignment = wrap_text
#     row_idx += 1
#
#     # Row 2: Scoring header
#     ws.cell(row=row_idx, column=2, value="Scoring")
#     ws.cell(row=row_idx, column=3, value="Diagnostic confidence")
#     ws.cell(row=row_idx, column=4, value="Tissue Visualization")
#     ws.cell(row=row_idx, column=5, value="Artifacts")
#     for col in [3, 4, 5]:
#         ws.cell(row=row_idx, column=col).alignment = wrap_text
#     row_idx += 1
#
#     # Row 3–4: RandID 0 and 1
#     for rid in [0, 1]:
#         ws.cell(row=row_idx, column=2, value=f"RandID={rid}")
#         for col in [3, 4, 5]:
#             ws.cell(row=row_idx, column=col, value="")
#             ws.cell(row=row_idx, column=col).fill = gray_fill
#             ws.cell(row=row_idx, column=col).alignment = wrap_text
#         row_idx += 1
#
# # Auto-adjust column widths
# for col in ws.columns:
#     max_len = max(len(str(cell.value)) if cell.value else 0 for cell in col)
#     ws.column_dimensions[col[0].column_letter].width = max(12, max_len + 2)
#
#
# # Save to Excel file
# wb.save(os.path.join(scan_roots[0],'ReaderSheet.xlsx'))