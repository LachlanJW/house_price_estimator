from sql_interpreter import get_data_from_sql


data = get_data_from_sql(col1="listingModel.price",
                         col2="listingModel.features.beds",
                         table='houses_test')
print(data)
