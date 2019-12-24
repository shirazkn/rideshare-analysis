# General statistical analysis methods to pre-analyze the data
# such as scatter-plots, correlations...

import json
from datetime import datetime, time
import numpy as np

# The oldest date in data
DATA_START_DATE = None


# *------------------------* DATA PREPARATION *------------------------* #


def show_null_columns(data):
    """
    Prints the columns in the dataset which have Null values
    :param data: Pandas DataFrame obj
    NOTE : There might be other columns with Null-like values, such as a string value "Unknown".
    These will not be counted.
    """
    null_entries = {c: data.isnull().sum()[c] for c in data.keys()}
    null_entries = {key: value for key, value in null_entries.items() if null_entries[key]}
    print(f"The following columns have Null values: ")
    print(json.dumps(null_entries, indent=4, sort_keys=True, default=str), "\n")


# *--------------------* DATE/TIME MANIPULATION *--------------------* #

def set_start_date(data_dates):
    global DATA_START_DATE
    DATA_START_DATE = data_dates.min()


def get_datetime_from_str(timestamp):
    """
    Returns datetime.datetime object from timestamp strings like 08/03/2019 08:15:00 PM
    """
    return datetime.strptime(timestamp, "%m/%d/%Y %I:%M:%S %p")


def get_days_in_data(dt):
    """
    Returns no. of days passed since start of data-set
    So <DATA_START_DATE, DATA_START_DATE+1, ... > are mapped to <0, 1, ... >
    """
    return (dt - DATA_START_DATE).days


def get_minutes_in_day(dt):
    """
    Returns the no. of minutes that transpired in the day
    """
    return dt.hour*60 + dt.minute


def get_day_in_week(dt):
    """
    Returns day of week, with 0: Monday and 6: Sunday
    """
    return dt.weekday()


def get_str_from_minutes(minutes):
    """
    Useful for plotting
    :param minutes: Output of get_minutes_in_day
    :return: str ("11:59 PM")
    """
    dt = time(int(np.floor(minutes / 60)), int(minutes % 60), 0)
    return time.strftime(dt, "%I:%M %p")


def change_xticks_to_time(plot, n_ticks=6):
    """
    Maps x-axis of a plot from Time (Minutes) to Time (%H:%M %p)
    """
    ticks = np.linspace(0, 1440, n_ticks)
    tick_labels = [get_str_from_minutes(t) for t in ticks[:-1]]
    tick_labels.append(tick_labels[0])
    plot.set_xticks(ticks)
    plot.set_xticklabels(tick_labels)


def minutes_in_day(points=1440):
    return np.linspace(0, 1440, points)


# *------------------------------* OTHER *------------------------------* #

def correlation_matrix(data, col_names):
    """
    Returns the correlation matrix between variables in <col_names>
    :param data: Pandas DataFrame obj
    :param col_names: Names of columns which you want to Analyze
    """
    corr_matrix = data.corr()
    for c in corr_matrix.columns:
        if c not in col_names:
            corr_matrix = corr_matrix.drop(c, axis=0).drop(c, axis=1)

    return corr_matrix
