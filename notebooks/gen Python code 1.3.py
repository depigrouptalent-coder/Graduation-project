import pandas as pd
import numpy as np
import uuid
import os
from datetime import datetime

# ==========================================
# 1. SIMULATION PARAMETERS & CONFIGURATION
# ==========================================
PROJECT_NAME = "NAC_Sustainable_Demographics"
START_YEAR = 2025
END_YEAR = 2050

# Demographics Target
AVG_FAMILY_SIZE = 3.9
TARGET_POP_DENSITY = 12.32  # Percentage
POPULATION_TARGET_2050 = 6500000
HOUSEHOLDS_TARGET_2050 = int(POPULATION_TARGET_2050 / AVG_FAMILY_SIZE) # ~1,666,666 households
HOUSEHOLDS_2026 = 400000

# File Size Management (~200MB limit allows for roughly 1M - 1.2M rows per file)
HOUSEHOLDS_PER_CHUNK = 200000 

OUTPUT_DIR = "NAC_Simulation_Data_All_Years"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demographics Probabilities 
GENDER = ['Male', 'Female']
RESIDENCE_TYPES = ['Apartment', 'Villa', 'Townhouse', 'Twin House']
RESIDENCE_SITES = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'CBD', 'Diplomatic District']

TRANSPORT_TYPES = ['LRT', 'Monorail', 'Electric Bus', 'Private Car', 'Walking/Bicycle']
TRANSPORT_PROBS = [0.25, 0.20, 0.15, 0.35, 0.05]

# Careers & Incomes (Based on Egyptian Platforms: Bayt, Paylab, Glassdoor, etc.)
CAREERS = {
    'Software Engineer / AI Spec': (180000, 600000),
    'Data Scientist': (200000, 700000),
    'Physician / Healthcare': (150000, 500000),
    'Civil/Architectural Engineer': (120000, 400000),
    'Financial Analyst / Banking': (140000, 450000),
    'Teacher / Educator': (80000, 180000),
    'Marketing / E-commerce': (100000, 350000),
    'Renewable Energy Tech': (120000, 300000),
    'Unemployed / Student / Child': (0, 0)
}

UNIVERSITY_MAJORS = ['Computer Science', 'Medicine', 'Engineering', 'Business Administration', 'Data Analytics', 'Renewable Energy', 'Education', 'None']
EDUCATION_TYPES = ['Public', 'Private', 'International', 'None']

HEALTH_CONDITIONS = [
    'Cardiovascular Services', 'Oncology Services', 'Neurology', 
    'Physical Therapy and Rehabilitation', 'Specialized Medical Councils', 
    'Mental Health', 'Ophthalmology and Surgeries', 'Specialized Dentistry', 'None'
]
HEALTH_PROBS = [0.03, 0.02, 0.02, 0.05, 0.02, 0.06, 0.05, 0.05, 0.70]

# ==========================================
# 2. HOUSEHOLD GENERATION LOGIC
# ==========================================
def get_career_and_income(age, is_student=False):
    if age < 22 or is_student or age > 65:
        return 'Unemployed / Student / Child', 0
    career = np.random.choice(list(CAREERS.keys())[:-1]) 
    min_inc, max_inc = CAREERS[career]
    return career, np.random.randint(min_inc, max_inc)

def generate_individual(year, role, hoh_age=None):
    if role == 'HoH_Male':
        age = np.random.randint(31, 65) 
        gender = 'Male'
        marital_status = 'Married'
    elif role == 'Spouse_Female':
        age = np.random.randint(25, 60) 
        gender = 'Female'
        marital_status = 'Married'
    elif role == 'Child':
        max_child_age = max(1, hoh_age - 25) if hoh_age else 20
        age = np.random.randint(0, min(max_child_age, 24))
        gender = np.random.choice(GENDER)
        marital_status = 'Single'
    elif role == 'Elderly':
        age = np.random.randint(65, 100)
        gender = np.random.choice(GENDER)
        marital_status = np.random.choice(['Married', 'Widowed'], p=[0.4, 0.6])
    else: 
        age = np.random.randint(22, 45)
        gender = np.random.choice(GENDER)
        marital_status = 'Single'

    birth_year = year - age
    life_expectancy = np.random.normal(74.5, 5) 
    death_year = birth_year + int(life_expectancy) if (birth_year + int(life_expectancy)) > year else np.nan
    
    edu_type = np.random.choice(EDUCATION_TYPES) if 6 <= age <= 22 else 'None'
    if 6 <= age <= 18:
        edu_grade = f"Grade {np.random.randint(1, 12)}"
    elif 18 < age <= 22:
        edu_grade = 'Undergraduate'
    else:
        edu_grade = 'Graduate/None'
        
    major = np.random.choice(UNIVERSITY_MAJORS) if age > 18 else 'None'
    career, income = get_career_and_income(age, is_student=(edu_grade in ['Undergraduate', 'Grade']))
    
    health_percent = np.random.randint(40, 100) if age > 65 else np.random.randint(75, 100)
    health_cond = np.random.choice(HEALTH_CONDITIONS, p=HEALTH_PROBS)
    needs_bed = 1 if health_cond in ['Cardiovascular Services', 'Oncology Services', 'Neurology'] and np.random.rand() > 0.6 else 0
    
    transport = np.random.choice(TRANSPORT_TYPES, p=TRANSPORT_PROBS)
    green_vehicle = 1 if transport == 'Private Car' and np.random.rand() > 0.7 else 0
    public_transit_green = 1 if transport in ['LRT', 'Monorail', 'Electric Bus'] else 0
    
    return {
        'Age': age, 'Year_of_Birth': birth_year, 'Year_of_Death_Projected': death_year, 
        'Gender': gender, 'Marital_Status': marital_status, 'Career': career, 'Income_Per_Year': income, 
        'Education_Grade': edu_grade, 'Education_Type': edu_type, 'University_Major': major, 
        'Health_Percent': health_percent, 'Needs_Hospital_Bed': needs_bed, 'Health_Condition': health_cond,
        'Transportation_Type': transport, 'Green_Fuel_Vehicle': green_vehicle, 
        'Public_Transit_Green_Fuel': public_transit_green
    }

def generate_household(year):
    re_id = str(uuid.uuid4().int)[:14] 
    residence_site = np.random.choice(RESIDENCE_SITES)
    res_type = np.random.choice(RESIDENCE_TYPES)
    
    solar = np.random.choice([0, 1], p=[0.75, 0.25])
    ac = np.random.choice([0, 1], p=[0.05, 0.95])
    
    base_elec_kw = np.random.randint(200, 600) - (100 if solar else 0)
    base_gas_m3 = np.random.randint(15, 40)
    
    members = []
    rand_val = np.random.rand()
    
    if rand_val < 0.70:
        hoh = generate_individual(year, 'HoH_Male')
        spouse = generate_individual(year, 'Spouse_Female')
        members.extend([hoh, spouse])
        
        num_children = np.random.choice([1, 2, 3, 4], p=[0.2, 0.45, 0.25, 0.1])
        for _ in range(num_children):
            members.append(generate_individual(year, 'Child', hoh_age=hoh['Age']))
            
        if np.random.rand() < 0.05:
            members.append(generate_individual(year, 'Elderly'))
            
    elif rand_val < 0.85:
        members.append(generate_individual(year, 'HoH_Male'))
        members.append(generate_individual(year, 'Spouse_Female'))
    else:
        members.append(generate_individual(year, 'Single'))

    family_size = len(members)
    
    elec_kw = base_elec_kw + (family_size * 50)
    gas_m3 = base_gas_m3 + (family_size * 5)
    water_liters = family_size * np.random.randint(3000, 5000) 
    food_kg = family_size * np.random.randint(25, 40)
    waste_kg = family_size * np.random.randint(15, 30) 
    smart_meter = np.random.choice([0, 1], p=[0.2, 0.8]) 
    
    energy_consumption = elec_kw + (gas_m3 * 10.5) 
    carbon_footprint = (elec_kw * 0.45) + (gas_m3 * 2.0) + (waste_kg * 0.5)
    
    rows = []
    for m in members:
        row = {
            'Simulation_Year': year,
            **m,
            'Real_Estate_ID': re_id,
            'Residence_Site': residence_site,
            'Type_of_Residence': res_type,
            'Family_Members_Count': family_size,
            'In_House_Solar_System': solar,
            'Air_Condition': ac,
            'Household_Electricity_Consumption_KW': elec_kw,
            'Household_Gas_Consumption_m3': gas_m3,
            'Household_Water_Consumption_Liters': water_liters,
            'Household_Food_Consumption_kg': food_kg,
            'Household_Waste_Generation_kg': waste_kg,
            'Household_Smart_Meter_Active': smart_meter,
            'Household_Monthly_Energy_Consumption': energy_consumption,
            'Household_Carbon_Footprint': carbon_footprint,
            'Target_Population_Density_Pct': TARGET_POP_DENSITY
        }
        rows.append(row)
        
    return rows

def get_target_households_for_year(year):
    """Calculates the target number of households for a given year using CAGR."""
    if year == 2025:
        return 376000 # Working backward ~6% from 2026
    elif year == 2026:
        return HOUSEHOLDS_2026
    else:
        # Calculate Compound Annual Growth Rate needed from 2026 to 2050
        cagr = (HOUSEHOLDS_TARGET_2050 / HOUSEHOLDS_2026) ** (1 / (2050 - 2026)) - 1
        years_since_2026 = year - 2026
        return int(HOUSEHOLDS_2026 * ((1 + cagr) ** years_since_2026))

# ==========================================
# 3. EXECUTION RUNNER (2025 to 2050)
# ==========================================
if __name__ == "__main__":
    print(f"Starting Multi-Year Simulation: {PROJECT_NAME} (2025 - 2050)")
    
    for current_year in range(START_YEAR, END_YEAR + 1):
        target_households = get_target_households_for_year(current_year)
        print(f"\n--- Processing Year: {current_year} | Target Households: {target_households:,} ---")
        
        households_generated = 0
        chunk_counter = 1
        
        while households_generated < target_households:
            # Determine how many households to generate in this specific chunk
            target_in_chunk = min(HOUSEHOLDS_PER_CHUNK, target_households - households_generated)
            
            chunk_data = []
            for _ in range(target_in_chunk):
                household_rows = generate_household(current_year)
                chunk_data.extend(household_rows)
                
            df_chunk = pd.DataFrame(chunk_data)
            
            # Save chunk to CSV
            file_name = f"NAC_Demographics_{current_year}_Part{chunk_counter}.csv"
            file_path = os.path.join(OUTPUT_DIR, file_name)
            df_chunk.to_csv(file_path, index=False)
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f" -> Saved {file_name} | Households: {target_in_chunk:,} | Rows: {len(df_chunk):,} | Size: {file_size_mb:.2f} MB")
            
            households_generated += target_in_chunk
            chunk_counter += 1

    print("\n✅ Simulation Complete. All 25 years processed and saved successfully.")
