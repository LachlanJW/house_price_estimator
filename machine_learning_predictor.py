import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import sql_interpreter as si
from loguru import logger as log

# Get dataframe and convert to csv
df = si.sql_query()
df.to_csv("processed_data.csv", index=False)

# Separate features and target variable
X = df[['lat', 'lng', 'beds', 'baths', 'parking']]
y = df['price']

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Save the model
model.save("house_price_model.h5")