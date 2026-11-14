import mysql.connector

import config

# Entry log joined with the vehicle owner, so the dashboard can show who came in.
ENTRY_SELECT = """
    SELECT e.log_id, e.plate_number, e.auth_result, e.confidence, e.time_stamp,
           o.full_name, o.category,
           TIMESTAMPDIFF(SECOND, e.time_stamp, NOW()) AS seconds_ago
    FROM ENTRY_LOG e
    LEFT JOIN REGISTERED_VEHICLE v ON v.plate_number = e.plate_number
    LEFT JOIN REGISTERED_OWNER o ON o.owner_id = v.owner_id
"""


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

    def _query(self, sql, params=()):
        conn = self.connect()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows

    def recent_entries(self, limit=5):
        """Returns the latest entries with owner details, newest first."""
        return self._query(f"{ENTRY_SELECT} ORDER BY e.time_stamp DESC LIMIT %s", (limit,))

    def day_summary(self, day):
        """Entry counts for one day (a datetime.date)."""
        row = self._query("""
            SELECT COUNT(*) AS total,
                   SUM(auth_result = 'Registered') AS registered,
                   SUM(auth_result = 'Visitor') AS visitors,
                   COUNT(DISTINCT plate_number) AS unique_vehicles
            FROM ENTRY_LOG WHERE DATE(time_stamp) = %s
        """, (day,))[0]
        return {key: int(value or 0) for key, value in row.items()}

    def hourly_counts(self, day):
        """Entries per hour of the day, split into Registered and Visitor (24 values each)."""
        rows = self._query("""
            SELECT HOUR(time_stamp) AS hour, auth_result, COUNT(*) AS n
            FROM ENTRY_LOG WHERE DATE(time_stamp) = %s
            GROUP BY hour, auth_result
        """, (day,))
        counts = {"Registered": [0] * 24, "Visitor": [0] * 24}
        for row in rows:
            if row["auth_result"] in counts:
                counts[row["auth_result"]][row["hour"]] = row["n"]
        return counts

    def search_entries(self, text="", status="", day=None, limit=500):
        """Entries matching a plate/name search, status and day; newest first."""
        conditions, params = [], []
        if text:
            conditions.append("(e.plate_number LIKE %s OR o.full_name LIKE %s)")
            params += [f"%{text}%", f"%{text}%"]
        if status:
            conditions.append("e.auth_result = %s")
            params.append(status)
        if day:
            conditions.append("DATE(e.time_stamp) = %s")
            params.append(day)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return self._query(f"{ENTRY_SELECT} {where} ORDER BY e.time_stamp DESC LIMIT %s", (*params, limit))

if __name__ == "__main__":
    # Test the database connection
    db = DatabaseHandler()
    res = db.authorize_vehicle("BJT8558")
    if res:
        print(f"Database Match: Welcome {res['full_name']} ({res['category']}, {res['faculty']})")
    else:
        print("Vehicle not found in registry.")
