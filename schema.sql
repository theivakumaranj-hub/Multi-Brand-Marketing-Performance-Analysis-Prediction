-- ========================================================
-- Marketing Campaign - Exploratory SQL Insights
-- Database: PostgreSQL
-- ========================================================

-- 1. Top 5 Best-Performing Campaigns by ROI
SELECT Campaign_ID, Campaign_Type, Revenue, Acquisition_Cost, ROI
FROM campaign_data
ORDER BY ROI DESC
LIMIT 5;

-- 2. Top 5 Worst-Performing Campaigns by ROI (Loss-Making)
SELECT Campaign_ID, Target_Audience, Revenue, Acquisition_Cost, ROI
FROM campaign_data
WHERE Profit_Flag = 0
ORDER BY ROI ASC
LIMIT 5;

-- 3. Average Revenue & Cost by Target Audience
SELECT Target_Audience, 
       ROUND(AVG(Revenue)::numeric, 2) AS Avg_Revenue, 
       ROUND(AVG(Acquisition_Cost)::numeric, 2) AS Avg_Cost
FROM campaign_data
GROUP BY Target_Audience
ORDER BY Avg_Revenue DESC;

-- 4. Overall Campaign Profitability Breakdown
SELECT Profit_Flag, 
       COUNT(Campaign_ID) AS Total_Campaigns,
       ROUND(AVG(ROI)::numeric, 4) AS Average_ROI
FROM campaign_data
GROUP BY Profit_Flag;

-- 5. Funnel Efficiency: Average Clicks to Conversions
SELECT ROUND(AVG(Impressions)::numeric, 0) AS Avg_Impressions,
       ROUND(AVG(Clicks)::numeric, 0) AS Avg_Clicks,
       ROUND(AVG(Conversions)::numeric, 0) AS Avg_Conversions
FROM campaign_data;