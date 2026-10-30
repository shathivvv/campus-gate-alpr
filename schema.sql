-- Database schema for the ALPR system (MySQL).
-- Reconstructed from the queries in db_handler.py: compare with your real tables
-- (phpMyAdmin > Export, structure only) and replace this file if they differ.

CREATE DATABASE IF NOT EXISTS umpsa_alpr;
USE umpsa_alpr;

CREATE TABLE IF NOT EXISTS REGISTERED_OWNER (
    owner_id    INT AUTO_INCREMENT PRIMARY KEY,
    full_name   VARCHAR(100) NOT NULL,
    category    VARCHAR(30)  NOT NULL,  -- e.g. Student, Staff
    faculty     VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS REGISTERED_VEHICLE (
    plate_number VARCHAR(15) PRIMARY KEY,
    owner_id     INT NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES REGISTERED_OWNER (owner_id)
);

CREATE TABLE IF NOT EXISTS ENTRY_LOG (
    log_id       INT AUTO_INCREMENT PRIMARY KEY,
    plate_number VARCHAR(15) NOT NULL,
    auth_result  VARCHAR(20) NOT NULL,  -- Registered / Visitor
    confidence   FLOAT,
    image_ref    VARCHAR(255),
    time_stamp   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_time_stamp (time_stamp)
);
