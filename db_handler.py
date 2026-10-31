import mysql.connector

import config


class DatabaseHandler:
    def __init__(self, db_config=None):
        self.db_config = db_config or config.DB_CONFIG

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

    def recent_entries(self, limit=5):
        """Returns the latest ENTRY_LOG rows, newest first."""
        conn = self.connect()
        cursor = conn.cursor(dictionary=True)

        cursor.execute(
            "SELECT plate_number, auth_result, time_stamp FROM ENTRY_LOG "
            "ORDER BY time_stamp DESC LIMIT %s",
            (limit,),
        )
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        return rows

if __name__ == "__main__":
    # Test the database connection
    db = DatabaseHandler()
    res = db.authorize_vehicle("BJT8558")
    if res:
        print(f"Database Match: Welcome {res['full_name']} ({res['category']}, {res['faculty']})")
    else:
        print("Vehicle not found in registry.")
