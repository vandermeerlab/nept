import numpy as np
import string
from collections import defaultdict


def read_file(file):
    contents = []
    temp = []

    this_file = open(file, 'r')
    lines = this_file.readlines()

    for line in lines:
        if line != '\n':
            temp.append(line)
        else:
            if len(temp) > 0:
                contents.append(temp)
                temp = []

    if len(temp) > 0:
        contents.append(temp)  # appends the last subject

    for i, content in enumerate(contents):
        contents[i] = ' '.join(content)

    this_file.close()

    return contents


def get_data(contents):

    header = {}
    copy = contents.split('\n')

    header_contents = dict(start_date='Start Date',
                           end_date=' End Date',
                           subject=' Subject',
                           experiment=' Experiment',
                           group=' Group',
                           box=' Box',
                           start_time=' Start Time',
                           end_time=' End Time',
                           program=' Program',
                           msn=' MSN')

    for line in copy:
        for key in header_contents:
            heading = line.split(':')
            if heading[0] == header_contents[key]:
                if key == 'start_time' or key == 'end_time':
                    header[key] = heading[1].lstrip() + ':' + heading[2] + ':' + heading[3]
                else:
                    header[key] = heading[1].lstrip()

    data = {}
    copy = contents.split()

    uppercase = string.ascii_uppercase

    idx = []
    for i, val in enumerate(copy):
        if val[0] in uppercase and val[1] == ':':
            idx.append(i)

    for i, j in zip(idx[:-1], idx[1:]):
        data[copy[i].lower()[0]] = [timestamp for timestamp in copy[i+1:j] if timestamp[-1] != ':']

    return header, data


def get_events(event_list):
    """Finds timestamps associated with each event.

    Parameters
    ----------
    event_times : list

    Returns
    -------
    events : dict
        With event type as key, timestamps as values.

    """
    float_events = [float(event) for event in event_list]
    active_events = [event for event in float_events if (event > 0.0)]

    events = defaultdict(list)

    for event in active_events:
        events[int(np.floor(event/10000))].append(event % 10000)

    return events


def get_subject(subject_content, data_key='b'):
    """Gets header and data from MedPC file for a single subject.
    subject_content: str
    data_key: str
        Default set to 'b'

    Returns
    -------
    header: dict
    data: list
    """
    (header, data) = get_data(subject_content)
    data = get_events(data[data_key])

    return header, data


def load_medpc(filename, f_assign_label):
    """Loads MedPC data file.

    Parameters
    ----------
    filename: MedPC file
    f_assign_label: module

    Returns
    -------
    rats_data: dict
        With each subject as keys. Contains dict of event as vdmlab.Epochs.

    """
    contents = read_file(filename)
    contents = contents[1:]

    rats_data = {}

    for content in contents:
        (header, data) = get_subject(content)
        rats_data[header['subject']] = f_assign_label(data)

    return rats_data