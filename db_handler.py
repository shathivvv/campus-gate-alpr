import mysql.connector
from datetime import datetime

class DatabaseHandler:
    def __init__(self, host="localhost", user="root", password="", database="umpsa_alpr"):
        self.db_config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database
        }

    def connect(self):
        return mysql.connector.connect(**self.db_config)

    def authorize_vehicle(self, plate_number):
        """Checks if a plate is registered and returns owner details."""
        conn = self.connect()
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT v.plate_number, o.full_name, o.category, o.faculty 
            FROM REGISTERED_VEHICLE v
            JOIN REGISTERED_OWNER o ON v.owner_id = o.owner_id
            WHERE v.plate_number = %s
        """
        cursor.execute(query, (plate_number,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return result

    def log_entry(self, plate_number, auth_result, confidence, image_ref=None):
        """Logs the vehicle entry into the ENTRY_LOG table."""
        conn = self.connect()
        cursor = conn.cursor()
        
        query = """
            INSERT INTO ENTRY_LOG (plate_number, auth_result, confidence, image_ref)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (plate_number, auth_result, float(confidence), image_ref))
        conn.commit()
        
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Test the database connection
    db = DatabaseHandler()
    res = db.authorize_vehicle("BJT8558")
    if res:
        print(f"Database Match: Welcome {res['full_name']} ({res['category']}, {res['faculty']})")
    else:
        print("Vehicle not found in registry.")