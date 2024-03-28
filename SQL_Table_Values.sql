-- MySQL dump 10.13  Distrib 8.0.36, for Win64 (x86_64)
--
-- Host: localhost    Database: houses
-- ------------------------------------------------------
-- Server version	8.0.36

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `houses_test`
--

DROP TABLE IF EXISTS `houses_test`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `houses_test` (
  `id` bigint DEFAULT NULL,
  `listingType` text,
  `listingModel.promoType` text,
  `listingModel.url` text,
  `listingModel.brandingAppearance` text,
  `listingModel.priceFromApm` tinyint(1) DEFAULT NULL,
  `listingModel.price` text,
  `listingModel.hasVideo` tinyint(1) DEFAULT NULL,
  `listingModel.address.street` text,
  `listingModel.address.suburb` text,
  `listingModel.address.state` text,
  `listingModel.address.postcode` text,
  `listingModel.address.lat` double DEFAULT NULL,
  `listingModel.address.lng` double DEFAULT NULL,
  `listingModel.features.beds` bigint DEFAULT NULL,
  `listingModel.features.baths` bigint DEFAULT NULL,
  `listingModel.features.parking` bigint DEFAULT NULL,
  `listingModel.features.propertyType` text,
  `listingModel.features.propertyTypeFormatted` text,
  `listingModel.features.isRural` tinyint(1) DEFAULT NULL,
  `listingModel.features.landSize` bigint DEFAULT NULL,
  `listingModel.features.landUnit` text,
  `listingModel.features.isRetirement` tinyint(1) DEFAULT NULL,
  `listingModel.inspection.openTime` text,
  `listingModel.inspection.closeTime` text,
  `listingModel.auction` text,
  `listingModel.tags.tagText` text,
  `listingModel.tags.tagClassName` text,
  `listingModel.displaySearchPriceRange` text,
  `listingModel.enableSingleLineAddress` tinyint(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `houses_test`
--

LOCK TABLES `houses_test` WRITE;
/*!40000 ALTER TABLE `houses_test` DISABLE KEYS */;
INSERT INTO `houses_test` VALUES (2018425198,'listing','standardpp','/g10-2-terry-connolly-street-coombs-act-2611-2018425198','dark',0,'$697,500',1,'G10/2 Terry Connolly Street','COOMBS','ACT','2611',-35.323605,149.04317,2,2,2,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 19 Feb 2024','is-sold',NULL,0),(2018738407,'listing','standardpp','/31-1-arthur-blakeley-way-coombs-act-2611-2018738407','light',0,'$690,000',0,'31/1 Arthur Blakeley Way','COOMBS','ACT','2611',-35.31613,149.03937,3,2,2,'Townhouse','Townhouse',0,123,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 08 Jan 2024','is-sold',NULL,0),(2018779337,'listing','standardpp','/14-120-john-gorton-drive-coombs-act-2611-2018779337','light',1,'$485,000',0,'14/120 John Gorton Drive','COOMBS','ACT','2611',-35.318466,149.03664,2,2,1,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 31 Jan 2024','is-sold',NULL,0),(2018799199,'listing','elite','/10-1-calaby-street-coombs-act-2611-2018799199','light',1,'$695,000',0,'10/1 Calaby Street','COOMBS','ACT','2611',-35.311756,149.03304,3,2,2,'Townhouse','Townhouse',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 22 Dec 2023','is-sold',NULL,0),(2018830616,'listing','standardpp','/33-2-newchurch-street-coombs-act-2611-2018830616','dark',0,'$538,000',0,'33/2 Newchurch Street','COOMBS','ACT','2611',-35.322235,149.04166,2,2,2,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 29 Jan 2024','is-sold',NULL,0),(2018886822,'listing','standardpp','/4a-harold-white-avenue-coombs-act-2611-2018886822','dark',0,'$735,000',0,'4A Harold White Avenue','COOMBS','ACT','2611',-35.324753,149.04477,3,2,2,'Townhouse','Townhouse',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 30 Jan 2024','is-sold',NULL,0),(2018887858,'listing','standardpp','/40-edgeworth-parade-coombs-act-2611-2018887858','dark',1,'$970,000',1,'40 Edgeworth Parade','COOMBS','ACT','2611',-35.31442,149.03783,4,3,2,'Townhouse','Townhouse',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 16 Jan 2024','is-sold',NULL,0),(2018944467,'listing','premiumplus','/23-60-john-gorton-drive-coombs-act-2611-2018944467','light',1,'$416,000',0,'23/60 John Gorton Drive','COOMBS','ACT','2611',-35.321484,149.04016,2,1,1,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 01 Feb 2024','is-sold',NULL,0),(2018957981,'listing','elite','/78-2-woodberry-avenue-coombs-act-2611-2018957981','dark',0,'$885,000',0,'78/2 Woodberry Avenue','COOMBS','ACT','2611',-35.324474,149.04715,4,2,2,'Townhouse','Townhouse',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 08 Feb 2024','is-sold',NULL,0),(2018983631,'listing','premiumplus','/101-2-newchurch-street-coombs-act-2611-2018983631','dark',0,'$527,000',0,'101/2 Newchurch Street','COOMBS','ACT','2611',-35.322235,149.04166,2,2,2,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 15 Feb 2024','is-sold',NULL,0),(2018992531,'listing','premiumplus','/32-96-arthur-blakeley-way-coombs-act-2611-2018992531','light',0,'$675,000',0,'32/96 Arthur Blakeley Way','COOMBS','ACT','2611',-35.31571,149.03575,3,2,2,'Townhouse','Townhouse',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 25 Jan 2024','is-sold',NULL,0),(2018994281,'listing','premiumplus','/46-30-pearlman-street-coombs-act-2611-2018994281','light',0,'$463,500',0,'46/30 Pearlman Street','COOMBS','ACT','2611',-35.324585,149.04811,2,1,1,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 16 Feb 2024','is-sold',NULL,0),(2018995719,'listing','premiumplus','/45-60-john-gorton-drive-coombs-act-2611-2018995719','light',0,'$433,500',0,'45/60 John Gorton Drive','COOMBS','ACT','2611',-35.321495,149.04015,2,1,2,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 20 Feb 2024','is-sold',NULL,0),(2018996265,'listing','premiumplus','/19-60-john-gorton-drive-coombs-act-2611-2018996265','light',0,'$425,000',1,'19/60 John Gorton Drive','COOMBS','ACT','2611',-35.321484,149.04016,2,1,1,'House','House',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 06 Mar 2024','is-sold',NULL,0),(2019006762,'listing','premiumplus','/71-1-cornelius-street-coombs-act-2611-2019006762','dark',0,'$585,000',1,'71/1 Cornelius Street','COOMBS','ACT','2611',-35.325283,149.04515,2,2,1,'Townhouse','Townhouse',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 23 Feb 2024','is-sold',NULL,0),(2019023583,'listing','premiumplus','/11-60-john-gorton-drive-coombs-act-2611-2019023583','light',0,'$422,000',0,'11/60 John Gorton Drive','COOMBS','ACT','2611',-35.321484,149.04016,2,1,1,'ApartmentUnitFlat','Apartment / Unit / Flat',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 07 Feb 2024','is-sold',NULL,0),(2019029242,'listing','elite','/32-castan-street-coombs-act-2611-2019029242','dark',0,'$1,485,000',1,'32 Castan Street','COOMBS','ACT','2611',-35.31777,149.04343,4,3,2,'House','House',0,0,'mÂ²',0,NULL,NULL,'2024-02-24T13:30:00','Sold at auction 24 Feb 2024','is-sold',NULL,0),(2019055203,'listing','premiumplus','/34-taggart-terrace-coombs-act-2611-2019055203','light',0,'$818,000',0,'34 Taggart Terrace','COOMBS','ACT','2611',-35.322968,149.04774,3,2,2,'Townhouse','Townhouse',0,0,'mÂ²',0,NULL,NULL,NULL,'Sold by private treaty 07 Mar 2024','is-sold',NULL,0),(2019070870,'listing','premiumplus','/9-redshaw-street-coombs-act-2611-2019070870','light',0,'$1,450,000',0,'9 Redshaw Street','COOMBS','ACT','2611',-35.313583,149.03995,4,3,2,'House','House',0,540,'mÂ²',0,NULL,NULL,'2024-03-16T10:00:00','Sold at auction 16 Mar 2024','is-sold',NULL,0),(2019073848,'listing','elite','/23-finemore-street-coombs-act-2611-2019073848','light',0,'$883,000',0,'23 Finemore Street','COOMBS','ACT','2611',-35.322765,149.0484,4,3,2,'House','House',0,199,'mÂ²',0,NULL,NULL,NULL,'Sold at auction 14 Mar 2024','is-sold',NULL,0);
/*!40000 ALTER TABLE `houses_test` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2024-03-28 13:23:31
