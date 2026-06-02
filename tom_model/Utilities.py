class Utilities:

    def __init__(self, schedule_file, table_file):


    def read_init_files(self, schedule_file, table_file, r32=False):

        schedule_filename = f'./Data/{schedule_file}.csv'
        table_filename = f'./Data/{table_file}.csv'

        match_schedule = pd.read_csv(schedule_filename, header=0, index_col=0)
        table = pd.read_csv(table_filename, header=0, index_col=0)

        return match_schedule, 