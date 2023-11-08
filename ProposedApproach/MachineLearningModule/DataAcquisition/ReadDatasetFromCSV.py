import pandas
import csv


def get_dataset_from_csv_file(file_path):
    """
    Function to read/get dataset from a CSV file.

    Input:
        file_path - Path to the CSV file

    Output:
        dataset
        class name
    """
    delimiter = get_csv_delimiter(file_path)
    dataset = pandas.read_csv(file_path, sep=delimiter, low_memory=False)
    class_name = list(dataset.columns)[-1]
    return dataset[:][:], class_name


def get_csv_delimiter(file_path):
    """
    Function to get the delimiter in the CSV file.

    Input:
        file_path - Path to the CSV file

    Output:
        delimiter in the CSV file
    """
    with open(file_path, 'r') as file:
        first_line = file.readline()
        dialect = csv.Sniffer().sniff(first_line)
        return str(dialect.delimiter)
