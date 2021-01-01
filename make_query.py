
def make_query_new(new_query, cols):

    import os
    import configparser
    from mysql.connector import connect
    import pandas as pd

    config = configparser.ConfigParser()
    config.read('config.ini')

    ## Connect to database
    db = connect(
        password=config['mysqlDB']['pass'],
        user=config['mysqlDB']['user'],
        host=config['mysqlDB']['host'],
        auth_plugin=config['mysqlDB']['auth_plugin'],
        database=config['mysqlDB']['db'])

    cursor = db.cursor()
    found = cursor.execute(new_query)
    found = cursor.fetchall()

    db.close()

    out = pd.DataFrame(found, columns = cols)

    return(out)
