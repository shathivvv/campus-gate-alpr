-- Database schema for the ALPR system (MySQL / MariaDB via XAMPP).
-- Exported from the umpsa_alpr database: mysql -u root < schema.sql

CREATE DATABASE IF NOT EXISTS umpsa_alpr;
USE umpsa_alpr;

CREATE TABLE IF NOT EXISTS registered_owner (
    owner_id   INT NOT NULL AUTO_INCREMENT,
    full_name  VARCHAR(100) NOT NULL,
    matric_no  VARCHAR(20) DEFAULT NULL,
    staff_no   VARCHAR(20) DEFAULT NULL,
    category   VARCHAR(20) DEFAULT NULL,  -- Student / Staff
    faculty    VARCHAR(50) DEFAULT NULL,
    contact    VARCHAR(20) DEFAULT NULL,
    PRIMARY KEY (owner_id)
);

CREATE TABLE IF NOT EXISTS registered_vehicle (
    plate_number   VARCHAR(15) NOT NULL,
    owner_id       INT DEFAULT NULL,
    vehicle_make   VARCHAR(50) DEFAULT NULL,
    vehicle_colour VARCHAR(30) DEFAULT NULL,
    sticker_no     VARCHAR(50) DEFAULT NULL,
    faculty        VARCHAR(50) DEFAULT NULL,
    contact        VARCHAR(20) DEFAULT NULL,
    PRIMARY KEY (plate_number),
    FOREIGN KEY (owner_id) REFERENCES registered_owner (owner_id)
);

CREATE TABLE IF NOT EXISTS entry_log (
    log_id       INT NOT NULL AUTO_INCREMENT,
    plate_number VARCHAR(15) DEFAULT NULL,
    time_stamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    auth_result  VARCHAR(20) DEFAULT NULL,  -- Registered / Visitor
    confidence   FLOAT DEFAULT NULL,
    image_ref    VARCHAR(255) DEFAULT NULL,
    PRIMARY KEY (log_id)
);
