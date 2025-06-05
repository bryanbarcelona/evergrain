import random #already there
import numpy as np #pip install numpy
from collections import Counter
from PIL import Image as imgr
from autocrop.autocrop import MultiPartImage, Background
import glob
from exif import Image #pip install exif
from string import ascii_uppercase
import os
from os import path
from PIL import Image
import re
import exiftool #pip install and add exiftool(-k).exe to site package and link in (exiftool.py class) - actually right here in the with statement opening exiftool
#from libxmp import XMPFiles, consts
import sys
import datetime
import glob
import pandas as pd


#log_file = open('D:\Projekte\Coding\\test_log.log', 'a')

# Creating a temporary work Folder



working_path = os.path.join(r"C:\Users\bryan\Desktop","temp")
if os.path.exists(working_path):
    print(working_path + ' : exists')
elif not os.path.exists(working_path):
    os.mkdir(working_path)

'''**********************************************************************************************************************************
++++++FUNCTIONS
This is where all the functions live:
- Getting a value within a given range with skewing and likelihood of min/max events
- Getting a capped standard normalized value within a range
- Organizing each of the groups in a batch based on syntax (this interprets the operational string)
- Distributing the allotted unique days within each group
**********************************************************************************************************************************'''

def get_weighted_value(min, max, skew, edge_likelyhood): # to get a pseudo standard normalized value within a range
    make_num_list = list(range(min, max+1))
    mean = (min + max + (len(make_num_list)/skew))/2

    a = (edge_likelyhood - 1)/pow(min - mean, 2)

    weights = []
    for i in range(len(make_num_list)):
        y = a * pow(make_num_list[i] - mean, 2) + 1
        y_integer = int(10*y)
        weights.append(y_integer)

    final_weight_list = []
    for i in range(len(make_num_list)):
        for j in range(weights[i]):
            final_weight_list.append(make_num_list[i])

    adjusted_max = random.choice(final_weight_list)
    return adjusted_max

def get_norm_dist_value(min, max, cap): # to get a capped standard normalized value within a range

    import random
    from math import e, sqrt, pi

    range_processed = list(range(min, max+1))
    
    if len(range_processed)==0:
        return 0
    elif len(range_processed)==1:
        result=range_processed[0]
        return result
    else:
        middle = int(sum(range_processed)/len(range_processed))
        std_dev = 0.25*(middle - range_processed[0])
        if std_dev == 0.0: std_dev = 1

        weights=[]
        #print(f"Range: {range_processed}, middle {middle} and Standard dev {std_dev}.")
        for i in range(len(range_processed)):

            g = int((e**(-1/2*((range_processed[i]-middle)/std_dev)**2)/std_dev*sqrt(2*pi))*100000)
            weights.append(g)

        result = random.choices(range_processed, weights=weights, k=1)[0]
        if result>cap: result = cap
        return result

def get_groups_organized(file_list, year, month, cmd_str): # organizing the individual groupings of files
    
    super_full_range = len(file_list) # dependent on total picture count
    #print(super_full_range)
    year = year #user input
    set_month = month #user input
    month_dict = {'Jan': 31, 'Feb': 28, 'Mar': 31, 'Apr': 30, 'May': 31, 'Jun': 30, 'Jul': 31, 'Aug': 31, 'Sep': 30, 'Oct': 31, 'Nov': 30, 'Dec': 31}
    num_month = month_dict.get(set_month)

    #**************************
    # Mapping the distance in the groups
    cmd_str = cmd_str #user input
    corrected_cmd_str = cmd_str.replace("{", "[").replace("}", "]").replace("][", "]|[")
    #print(cmd_str)
    #print(corrected_cmd_str)
    super_groups = corrected_cmd_str.split("|")


    #figuring out all the homogenous ranges
    homo_range_map = []
    homo_range_map_hc = []
    for elements in range(len(super_groups)):
        min_range = int(super_groups[elements].split("[")[1].split("-")[0])
        max_range = int(super_groups[elements].split("-")[1].split("]")[0])
        homo_range_map.append(max_range-min_range+1)
        homo_range_map_hc.append(max_range-min_range+1)
    homo_range_map.append(0)
    homo_range_map_hc.append(0)
    #print(homo_range_map)

    mapped = [['init', 0]] # initializing the series to first picture

    #figuring out all the diverse ranges
    for elements in range(len(super_groups)):
        min_range = super_groups[elements].split("[")[1].split("-")[0]
        max_range = super_groups[elements].split("-")[1].split("]")[0]
        mapped.append([int(min_range), int(max_range)])
    mapped.append([super_full_range+1, 'exit'])

    div_range_map=[]
    div_range_map_hc=[]
    for i in range(len(mapped)-1):
        cur_range = mapped[i+1][0] - mapped[i][1] - 1
        div_range_map.append(cur_range)
        div_range_map_hc.append(cur_range)


    min_num_of_unique_days = sum(map(lambda x : x != 0, homo_range_map)) + sum(map(lambda x : x != 0, div_range_map))
    max_num_of_unique_days = sum(div_range_map) + len(homo_range_map) - 1 #to adjust for the always leading empty group for homo

    
    #homo_range_map_hc = homo_range_map

    #print(mapped)
    #print(div_range_map)
    #print(f"Allowed are at least {min_num_of_unique_days} and at most {max_num_of_unique_days} of unique days.")


    choice_unique_days = get_norm_dist_value(min_num_of_unique_days, max_num_of_unique_days, num_month)
    list_of_month_days = list(range(1, num_month+1))
    day_set = random.sample(list_of_month_days, choice_unique_days)
    day_set.sort()
    '''
    for i in range(choice_unique_days):
        picked_day = random.sample(list_of_month_days)
        day_set.append(picked_day)
    day_set.sort()
    '''
    controller = choice_unique_days
    #print(choice_unique_days)
    map_unique_day_distribution = []
    allowed_choice_of_unique_days = choice_unique_days - min_num_of_unique_days + 1 # to adjust for the group already in
    
    check_to_see_if_prev_zero = []
    
    for i in range(len(div_range_map)):
        #print(f"Iteration: {i}")
        absorbable_min = choice_unique_days - sum(map(lambda x : x != 0, homo_range_map)) - (sum(div_range_map)-div_range_map[i])
        #print(f"Before correction: The most: {allowed_choice_of_unique_days}. At least: {absorbable_min}. left from total: {choice_unique_days}")
        if allowed_choice_of_unique_days > div_range_map[i]: allowed_choice_of_unique_days=div_range_map[i]
        #if allowed_choice_of_unique_days <= 0 and check_to_see_if_prev_zero[i]==0: allowed_choice_of_unique_days=1
        
        if absorbable_min <= 0 and div_range_map[i] == 0:
            absorbable_min = 0
        elif absorbable_min <= 0 and div_range_map[i] != 0:
            absorbable_min = 1
        elif absorbable_min > allowed_choice_of_unique_days:
            absorbable_min = allowed_choice_of_unique_days


        #print(f"The most: {allowed_choice_of_unique_days}. At least: {absorbable_min}. left from total: {choice_unique_days}")
        choice_within_group = get_norm_dist_value(absorbable_min, allowed_choice_of_unique_days, allowed_choice_of_unique_days)
        #print(f"I chose: {choice_within_group}")
        map_unique_day_distribution.append(choice_within_group)
        check_to_see_if_prev_zero.append(div_range_map[i])
        homo_range_map[i] = 0
        div_range_map[i] = 0
        choice_unique_days = choice_unique_days - (choice_within_group + 1) # dont forget the static "homo" group
        allowed_choice_of_unique_days = choice_unique_days - (sum(map(lambda x : x != 0, homo_range_map)) + sum(map(lambda x : x != 0, div_range_map)) -1) #minus one because we are already inside that "in between" group as it hasnt been deleted yet
        #print(homo_range_map, div_range_map, choice_unique_days, check_to_see_if_prev_zero)
        

    
    used_unique_days = (len(map_unique_day_distribution)-1)+sum(map_unique_day_distribution)
    #print(used_unique_days)
    to_be_distributed = controller - used_unique_days
    #print(to_be_distributed)
    #print(div_range_map_hc)
    diff_in_open_slots_and_dist = []
    #print(map_unique_day_distribution)
    for i in range(to_be_distributed):
        diff_in_open_slots_and_dist = []
        for j in range(len(map_unique_day_distribution)):
            diff_in_open_slots_and_dist.append(div_range_map_hc[j]-map_unique_day_distribution[j])
        #print(diff_in_open_slots_and_dist)
  
        max_value = max(diff_in_open_slots_and_dist)
        indices = [p for p, x in enumerate(diff_in_open_slots_and_dist) if x == max_value]
        picked_index = random.choice(indices)
        map_unique_day_distribution[picked_index]=map_unique_day_distribution[picked_index]+1
    
    #print(map_unique_day_distribution)

    return map_unique_day_distribution, div_range_map_hc, homo_range_map_hc, controller, day_set

def dist_inside_group(full_range, unique_groups, homo_list, days_set, controller): # assigning unique value within each grouping
    controller = controller
    allowable_range = full_range - unique_groups+1

    for j in range(unique_groups):
        
        
        if unique_groups == 1:
            current_range = allowable_range
        else:
            current_range = get_weighted_value(1,allowable_range, -0.1, 0.3)
        print(f"From an allowed range of {allowable_range} I chose a length of {current_range}.")
        
        for i in range(current_range):
            day_filename_list.append(str(days_set[controller]).zfill(2))
            print(f"Added {days_set[controller]}. My controller is at {controller}.")
        controller=controller+1
        print(f"Current range is {current_range} from a maximal length of {allowable_range}")
        print(f"The unique group is: {unique_groups}.")
        

        unique_groups = unique_groups-1
        full_range = full_range - current_range
        print(f"I have lowered the numer of left unique subgroups to {unique_groups}. The leftover range now is {full_range}.")
        allowable_range = full_range - unique_groups+1
    
    for homo in range(homo_list):
        day_filename_list.append(str(days_set[controller]).zfill(2))
        print(f"Added {days_set[controller]}. I am homo. My controller is at {controller}.")
    controller=controller+1
    return controller, day_filename_list

def get_usb_drive(): # get correct USB device with scanned picture. The USB drive needs the hook.hook file in the top most directory
    for drive in ascii_uppercase:
        if path.exists(f"{drive}:\\hook.hook"):
            return drive + ":\\"   

'''**********************************************************************************************************************************
++++++PROCESSING SCANS
The following section first lists all scanned items. It then loads a background reference to know what the background of the scan is.
All scans are then looped through to cut out all the sub images.These are then labeled in such a way that they stay in order. Finally,
all the pictures are trimmed on each side as there is always a little bit of unclean edges.
**********************************************************************************************************************************'''
xlsx = 'D:\\Projekte\\Coding\\Photo crop and smart renamer\\Date_instructions.xlsx'

drive = get_usb_drive()
print(f"USB with scans was detected in {drive}")
#scanned_raw_image_list = glob.glob(f'C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\Unfinished Scans (need further details from Meldorf)\\Kevin Malente\\EPSON*.JPG')
scanned_raw_image_list = glob.glob(f'{drive}\\EPSCAN\\001\\EPSON*.JPG')
scanned_raw_image_list.sort()
print(f"There are {len(scanned_raw_image_list)} scanned items.")
#print(scanned_raw_image_list)
# A saved scan with the scanner empty.
blank_img = imgr.open('D:\\Projekte\\Coding\\Photo crop and smart renamer\\background_reference.jpg')
background = Background().load_from_image(blank_img, dpi=600)

for i in range(len(scanned_raw_image_list)):   # cutting out multiple images from one scan

    # A saved scan with multiple photos loaded in the scanner
    scan_img = imgr.open(scanned_raw_image_list[i])
    scan = MultiPartImage(scan_img, background, dpi=600)

    for index, photo in enumerate(scan):
        photo.save(f"C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\temp\\{str(i).zfill(3)}image-{index}.jpg", dpi=(600, 600), quality=90)

cut_image_list = glob.glob('C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\temp\\*image-*.jpg')
print(f"There are {len(cut_image_list)} cut out items.")
#print(cut_image_list)
for i in range(len(cut_image_list)):  # cropping the edges
    with imgr.open(cut_image_list[i]) as im:

        width, height = im.size
        (left, upper, right, lower) = (15, 15, width - 15, height - 15)

        im_crop = im.crop((left, upper, right, lower))
        #im_crop.save(f"D:\\Test Folder\\Dont\\{str(i).zfill(3)}image-{index}.jpg", dpi=(600, 600), quality=90)
        im_crop.save(cut_image_list[i], dpi=(600, 600), quality=91)
cut_image_list.sort()
'''**********************************************************************************************************************************
++++++USER INPUT

**********************************************************************************************************************************'''
cut_image_list = glob.glob('C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\temp\\*image-*.jpg')
cut_image_list.sort()
#cut_image_list = ["Platform", "1&1", "1&1", "Adobe ID", "Amazon", "AOK", "Apple", "BioRender", "Bonavendi", "Booking.com", "Buffl", "Celemony", "ChemAxon", "Deezer", "DriveNow", "Dropbox", "Ebay", "ebay", "Elster", "Elster", "Platform", "1&1", "1&1", "Adobe ID", "Amazon", "AOK", "Apple", "BioRender", "Bonavendi", "Booking.com", "Buffl", "Celemony", "ChemAxon", "Deezer", "DriveNow", "Dropbox", "Ebay", "ebay", "Elster", "Elster"] #40
#op_str = "[4-7]"
#year = 1995
#month = "May"

'''
op_str = input("Please enter the operational command. [x-y] denote ranges within one day, | denote separation between ranges.")
year = input("What year are these pictures from?")
month = input("What month are these pictures from?")
day = input("What day are these pictures from? If unknown enter \"NONE\".")
month_dict = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
month_numerical = month_dict.get(month)

# CITY
city= input("Which city are these pictures from?")
city_cmd= f"-LocationCreatedCity={city}"
loc_pic = city_cmd.encode('utf-8')

# EVENT
event= input("What event is this for?")
event_cmd = f"-Event={event}"
event_pic = event_cmd.encode('utf-8')

'''

'''**********************************************************************************************************************************
++++++ASSIGNING DATES FOR PHOTOS
Here is where the decision is made how to assign the Metadata - particularly dates - to all the pictures in this batch. Month and
year are not flexible as of now. Days are optionally flexible 
**********************************************************************************************************************************'''

'''
if day != "NONE":
    scan_circumstance_xp_comment = b"-XPComment=Date based on known date. Time approximated."
    scan_circumstance_user_comment = b"-UserComment=Date based on known date. Time approximated."
    day = str(day).zfill(2)
    every_second=[]
    for hour in range(24):
        for min in range(60):
            for sec in range(60):
                every_second.append([f"{str(hour).zfill(2)}{str(min).zfill(2)}{str(sec).zfill(2)}", f"{str(hour).zfill(2)}:{str(min).zfill(2)}:{str(sec).zfill(2)}"])

    selected_timeframe=[]
    for i in range(32400, 64800):
        selected_timeframe.append(every_second[i])

    my_picks=[]
    my_picks=random.sample(selected_timeframe, k=len(cut_image_list))
    my_picks.sort()

    master_filename_list=[]
    master_EXIF_date_list=[]
    for i in range(len(cut_image_list)):
        master_filename_list.append(f"{year}{month_numerical}{day} {my_picks[i][0]} {event}.jpg")
        master_EXIF_date_list.append(f"{year}:{month_numerical}:{day} {my_picks[i][1]}")
    #print(master_filename_list)
    #print(master_EXIF_date_list)
else:
    scan_circumstance_xp_comment = b"-XPComment=Date based on known year and month. Day and time approximated."
    scan_circumstance_user_comment = b"-UserComment=Date based on known year and month. Day and time approximated."
    dist_het, len_het, len_hom, controller_end, day_set = get_groups_organized(cut_image_list, year, month, op_str)

    day_filename_list=[]
    controller = 0

    for i in range(len(dist_het)):
        controller, day_filename_list = dist_inside_group(len_het[i], dist_het[i], len_hom[i], day_set, controller) 

    #print(day_filename_list)

    every_second=[]
    for hour in range(24):
        for min in range(60):
            for sec in range(60):
                every_second.append([f"{str(hour).zfill(2)}{str(min).zfill(2)}{str(sec).zfill(2)}", f"{str(hour).zfill(2)}:{str(min).zfill(2)}:{str(sec).zfill(2)}"])

    selected_timeframe=[]
    for i in range(32400, 64800):
        selected_timeframe.append(every_second[i])

    my_picks=[]
    my_picks=random.sample(selected_timeframe, k=len(day_filename_list))
    my_picks.sort()

    master_filename_list=[]
    master_EXIF_date_list=[]
    for i in range(len(day_filename_list)):
        master_filename_list.append(f"{year}{month_numerical}{day_filename_list[i]} {my_picks[i][0]} {event}.jpg")
        master_EXIF_date_list.append(f"{year}:{month_numerical}:{day_filename_list[i]} {my_picks[i][1]}")
    print(master_filename_list)
    print(master_EXIF_date_list)
'''

'''**********************************************************************************************************************************
++++++ASSIGNING DATES FOR PHOTOS - Excel Alternative
**********************************************************************************************************************************'''

date_instructions_table = pd.read_excel(xlsx, sheet_name=0) # can also index sheet by name or fetch all sheets
master_filename_list = date_instructions_table['Filename'].tolist()
master_EXIF_date_list = date_instructions_table['DateStamp'].tolist()
loc_pic = date_instructions_table['Location'].tolist()
event_pic = date_instructions_table['Event'].tolist()
unique_tags_pic = date_instructions_table['Tags'].tolist()
date_instructions_table = pd.read_excel(xlsx, sheet_name=1)
pic_tags = date_instructions_table['Tags'].tolist()
scan_circumstance_xp_comment = b"-XPComment=Date based on known year and month. Day and time approximated."
scan_circumstance_user_comment = b"-UserComment=Date based on known year and month. Day and time approximated."

if len(master_filename_list) == len(cut_image_list) and len(master_EXIF_date_list) == len(cut_image_list) and len(loc_pic) == len(cut_image_list) and len(event_pic) == len(cut_image_list):
    print("Lists are of appropriate length. Continuing.")
    pass
else:
    print("Lists are not of equal length. Ending program.")
    sys.exit(0)

#file_names=['19950505 113203.jpg', '19950506 114713.jpg', '19950506 124131.jpg', '19950519 130925.jpg', '19950519 140752.jpg', '19950519 142047.jpg', '19950519 143911.jpg', '19950525 145510.jpg', '19950531 153205.jpg', '19950531 163510.jpg', '19950531 171749.jpg']
#datetimes= ['1995:05:05 11:32:03', '1995:05:06 11:47:13', '1995:05:06 12:41:31', '1995:05:19 13:09:25', '1995:05:19 14:07:52', '1995:05:19 14:20:47', '1995:05:19 14:39:11', '1995:05:25 14:55:10', '1995:05:31 15:32:05', '1995:05:31 16:35:10', '1995:05:31 17:17:49']


for i in range(len(cut_image_list)):
    current_datetime = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    scan_date_str = f"-ScanDate={current_datetime}"
    scan_date_bytes = scan_date_str.encode('utf-8')

    #string = "D:\\Test Folder\\XMP\\000image-3.jpg"
    file_path = cut_image_list[i].encode('utf-8')
    renamed_file = f'C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\temp\\{master_filename_list[i]}'

    digi_datetime_cmd = f"-CreateDate={master_EXIF_date_list[i]}"
    digitize_date = digi_datetime_cmd.encode('utf-8')

    taken_day_str = f"-DateTimeOriginal={master_EXIF_date_list[i]}"
    taken_day_bytes = taken_day_str.encode('utf-8')

    taken_day_str_exif = f"-EXIF:DateTimeOriginal={master_EXIF_date_list[i]}"
    taken_day_bytes_exif = taken_day_str_exif.encode('utf-8')

    # CITY
    city_cmd= f"-LocationCreatedCity={loc_pic[i]}"
    loc_pic_bytes = city_cmd.encode('utf-8')

    # EVENT
    event_cmd = f"-Event={event_pic[i]}"
    event_pic_bytes = event_cmd.encode('utf-8')

    keywords = str(unique_tags_pic[i]).split("; ")
    
    with exiftool.ExifTool("C:\\Program Files\\Python310\\Scripts\\exiftool-12.36\\exiftool(-k).exe") as et:
    #with exiftool.ExifTool("C:\\Users\\Bryan Barcelona\\Downloads\\exiftool-12.36\\exiftool(-k).exe") as et:
        et.execute(scan_circumstance_xp_comment, file_path)
        et.execute(scan_circumstance_user_comment, file_path)
        et.execute(taken_day_bytes, file_path)
        et.execute(taken_day_bytes_exif, file_path)
        et.execute(digitize_date, file_path)
        et.execute(scan_date_bytes, file_path)
        et.execute(loc_pic_bytes, file_path)
        et.execute(event_pic_bytes, file_path)

        if keywords[0] != "nan":        
            for p in range(len(keywords)):
                keyword = f"-Subject+={keywords[p]}"
                keyword_bytes = keyword.encode('utf-8')
                et.execute(keyword_bytes, file_path)
        else:
            pass
            
        for j in range(len(pic_tags)):
            keyword = f"-Subject+={pic_tags[j]}"
            keyword_bytes = keyword.encode('utf-8')
            et.execute(keyword_bytes, file_path)


        et.terminate()
    
    os.rename(cut_image_list[i], renamed_file)

delete_copy = glob.glob('C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\temp\\*.jpg_original')

for filePath in delete_copy:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)


delete_copy = glob.glob(f'{drive}\\EPSCAN\\001\\EPSON*.JPG')

for filePath in delete_copy:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)


# Renaming the temporary folder to its final folder
current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

oldpath = "C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\temp"
newpath = f"C:\\Users\\Bryan Barcelona\\Desktop\\Bryan\\2022 Scan Project\\{current_datetime}"

if os.path.exists(oldpath):
    os.rename(oldpath, newpath)
