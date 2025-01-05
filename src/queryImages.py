import psycopg
import cv2
import numpy as np

# Function to convert binary data to image
def convert_binary_to_image(binary_data):
    # Convert binary data back to image using numpy and OpenCV
    np_array = np.frombuffer(binary_data, dtype=np.uint8)  # Convert binary to numpy array
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Decode the numpy array to an image
    return image

# Function to select and process images from the database
def fetch_images_from_db(connection):
    # SQL query to select the image data from the table
    query = "SELECT image FROM game_camera"

    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()  # Fetch all rows from the result
        image_id = 0
        for row in rows:
            image_id += 1
            binary_data = row[0]  # Binary image data from the 'frame_data' column
            
            # Convert binary data to image
            image = convert_binary_to_image(binary_data)
            
            if image is not None:
                # Aave the image as a file
                cv2.imwrite(f"retrieved_image_{image_id}.jpg", image)
                print(f"Image with ID {image_id} processed and saved.")
                
                # Wait for a key press to move to the next image
                cv2.waitKey(0)

        # Close the OpenCV windows once all images are processed
        cv2.destroyAllWindows()

# Database connection parameters
host = '<HOST>'
port = '<PORT>'
dbname = '<DBNAME>'
user = '<USER>'
password = '<PASSWORD>'

# Connect to the PostgreSQL database
with psycopg.connect(host=host, port=port, dbname=dbname, user=user, password=password) as conn:
    # Fetch and process images from the database
    fetch_images_from_db(conn)

