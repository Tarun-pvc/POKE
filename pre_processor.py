"""
Addressed in normalization:
- Roman Numerals
- Fractions
- Ratios
- Time 
- Telephone Numbers
- Hindi Numbers
"""

import os
import re


def normalize_text(text):

    # Normalize Roman numerals
    def roman_to_int(match):
        roman_numeral = match.group(0)
        roman_dict = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        result = 0
        prev_value = 0
        for char in reversed(roman_numeral):
            value = roman_dict[char]
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value
        return str(result)

    roman_pattern = r'\b[IVXLCDM]+\b'
    text = re.sub(roman_pattern, roman_to_int, text)

    # Normalize fractions (ex: 1/2 to ½)

    def fraction_to_unicode(match):
        fraction = match.group(0)
        fraction_dict = {
            '1/2': '½', '1/3': '⅓', '2/3': '⅔',
            '1/4': '¼', '3/4': '¾', '1/5': '⅕',
            '2/5': '⅖', '3/5': '⅗', '4/5': '⅘',
            '1/6': '⅙', '5/6': '⅚', '1/7': '⅐',
            '1/8': '⅛', '3/8': '⅜', '5/8': '⅝',
            '7/8': '⅞'
        }
        return fraction_dict.get(fraction, fraction)

    fraction_pattern = r'\b\d+/\d+\b'
    text = re.sub(fraction_pattern, fraction_to_unicode, text)

    ratio_pattern = r'\b\d+:\d+\b'
    text = re.sub(ratio_pattern, lambda x: x.group(0), text)

    # ignore
    def remove_zeros(decimal_str):
        cleaned_str = decimal_str.lstrip('0').rstrip('0').rstrip('.')
        if not cleaned_str:
            cleaned_str = '0'

        return cleaned_str

    # Normalize time (ex: 2:30 PM and 2:30PM to 14:30)

    def normalize_time(match):
        time_str = match.group(0)
        try:
            from datetime import datetime
            formatted_time = datetime.strptime(
                re.sub(r'\s', '', time_str), '%I:%M%p').strftime('%H:%M')
            return formatted_time
        except ValueError:
            return time_str

    time_pattern = r'\b\d{1,2}:\d{2}(?: ?[APap][Mm])?\b'
    text = re.sub(time_pattern, normalize_time, text)

    # Normalize Telephone Numbers (ex: (123) 456-7890 to 123-456-7890)

    def normalize_phone(match):
        phone_str = match.group(0)
        return re.sub(r'\(|\)', '', phone_str)

    phone_pattern = r'\(\d{3}\) \d{3}-\d{4}'
    text = re.sub(phone_pattern, normalize_phone, text)

    # Normalize dates ex , 12/31/2022 to 2022-12-31
    def normalize_date(match):
        date_str = match.group(0)
        try:
            from datetime import datetime
            formatted_date = datetime.strptime(
                date_str, '%m/%d/%Y').strftime('%Y-%m-%d')
            return formatted_date
        except ValueError:
            return date_str

    date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
    text = re.sub(date_pattern, normalize_date, text)

    # Normalize Year (ex: 1990 to 1990)
    year_pattern = r'\b\d{4}\b'
    text = re.sub(year_pattern, lambda x: x.group(0), text)

    # NOrmalize Numbers (ex: १७ to 17)
    def normalize_hindi_numbers(text):
        hindi_numbers = {
            '०': '0',
            '१': '1',
            '२': '2',
            '३': '3',
            '४': '4',
            '५': '5',
            '६': '6',
            '७': '7',
            '८': '8',
            '९': '9'
        }

        normalized_text = ''.join(
            hindi_numbers[char] if char in hindi_numbers else char for char in text)

        return normalized_text

    text = normalize_hindi_numbers(text)

    return text


def write_to_file(content, folder_path, file_name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


# input_folder_path = "./annotated_hindi_data/data"
# output_folder_path = "./processed_data/"

input_folder_path = './hindi_dump/'
output_folder_path = './processed_data_noisy/'


def iterate_through_txt_files(folder_path, n):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    i = 0
    for txt_file in txt_files:
        if i == n:
            return
        try:
            file_path = os.path.join(folder_path, txt_file)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                content = normalize_text(content)
                file_name = os.path.splitext(txt_file)[0]
                print('file name: ', file_name)
                write_to_file(content, output_folder_path,
                              f"processed_{file_name}.txt")
                i += 1
        except:
            print('issue')


iterate_through_txt_files(input_folder_path, 50)
