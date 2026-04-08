import pandas as pd
import numpy as np
import uuid
import os

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
PROJECT_NAME = "NAC_Sustainable_Demographics_2025_2050"
START_YEAR = 2025
END_YEAR = 2050

# Target population logic
# 2026 Target: 400,000 residences. 2050 Target: 6.5M people (~1.66M residences at 3.9 avg size)
AVG_FAMILY_SIZE = 3.9
TARGET_POP_DENSITY = 0.1232 
ROWS_PER_CHUNK = 250000  # Keeps files safely under 100 MB limit
NOISE_RATIO = 0.03

OUTPUT_DIR = "NAC_Simulated_Data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Demographics
RESIDENCE_TYPES = ['Apartment', 'Villa', 'Townhouse', 'Twin House']
RESIDENCE_SITES = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'CBD', 'Diplomatic District']
TRANSPORT_TYPES = ['LRT', 'Monorail', 'Electric Bus', 'Private Car', 'Walking/Bicycle']
TRANSPORT_PROBS = [0.25, 0.20, 0.15, 0.35, 0.05]

# Income per year in EGP (Based on Bayt, Glassdoor, Paylab Egypt 2025 projections)
CAREERS = {
    'Software Engineer / AI Specialist': (180000, 600000),
    'Data Scientist': (200000, 700000),
    'Medical Doctor / Surgeon': (150000, 500000),
    'Civil / Urban Engineer': (120000, 400000),
    'Financial Analyst / Investment': (140000, 450000),
    'Renewable Energy Technician': (90000, 250000),
    'Educator / Professor': (80000, 200000),
    'Nurse / Healthcare Worker': (70000, 150000),
    'Unemployed / Student / Child': (0, 0)
}

MAJORS = ['Computer Science', 'Medicine', 'Engineering', 'Business', 'Data Analytics', 'Nursing', 'Education', 'None']
EDU_TYPES = ['Public', 'Private', 'International', 'None']

HEALTH_CONDS = [
    'Cardiovascular Services', 'Oncology Services', 'Neurology', 
    'Physical Therapy and Rehabilitation', 'Specialized Medical Councils', 
    'Mental Health', 'Ophthalmology and Surgeries', 'Specialized Dentistry', 'None'
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def inject_noise(df):
    """Injects exactly up to 3% noise for ETL pre-processing tests."""
    num_rows = len(df)
    noise_count = int(num_rows * NOISE_RATIO)
    
    # 1. Missing Values
    for col in ['Income_Per_Year_EGP', 'Electricity_Consumption_KW', 'Carbon_Footprint_kg']:
        nan_idx = np.random.choice(df.index, size=int(noise_count/3), replace=False)
        df.loc[nan_idx, col] = np.nan
        
    # 2. Outliers (Impossible ages)
    outlier_idx = np.random.choice(df.index, size=int(noise_count/3), replace=False)
    df.loc[outlier_idx, 'Age'] = np.random.randint(200, 999, size=len(outlier_idx))
    
    # 3. Text Casing Issues
    case_idx = np.random.choice(df.index, size=int(noise_count/3), replace=False)
    if 'Gender' in df.columns:
        df.loc[case_idx, 'Gender'] = df.loc[case_idx, 'Gender'].apply(lambda x: str(x).upper() if pd.notnull(x) else x)
        
    return df

def generate_households_for_year(year):
    """Determines how many households to generate based on year interpolation."""
    if year <= 2026:
        return 400000
    else:
        # Linearly interpolate from 400,000 (2026) to 1,666,666 (2050) to reach 6.5M people
        return int(np.interp(year, [2026, 2050], [400000, 1666666]))

# ==========================================
# 3. MAIN VECTORIZED GENERATOR
# ==========================================
def generate_yearly_data(year, total_households):
    print(f"[{year}] Generating {total_households} households...")
    
    # 1. Define Households
    # We use probabilities to get an average family size of ~3.9
    family_sizes = np.random.choice([1, 2, 3, 4, 5, 6], size=total_households, p=[0.05, 0.15, 0.20, 0.35, 0.15, 0.10])
    
    household_df = pd.DataFrame({
        'Real_Estate_ID': [str(uuid.uuid4().int)[:14] for _ in range(total_households)],
        'Residence_Site': np.random.choice(RESIDENCE_SITES, total_households),
        'Type_of_Residence': np.random.choice(RESIDENCE_TYPES, total_households),
        'Family_Members': family_sizes,
        'In_House_Solar_System': np.random.choice([0, 1], total_households, p=[0.6, 0.4]),
        'Air_Condition': np.random.choice([0, 1], total_households, p=[0.1, 0.9]),
        'Electricity_Consumption_KW': np.random.randint(200, 1500, total_households),
        'Gas_Consumption_m3': np.random.randint(10, 80, total_households),
        'Water_Consumption_Liters': np.random.randint(4000, 15000, total_households),
        'Food_Consumption_kg': np.random.randint(50, 200, total_households),
        'Solid_Waste_kg': np.random.randint(30, 100, total_households), # Added for sustainability
        'Transportation_Type': np.random.choice(TRANSPORT_TYPES, total_households, p=TRANSPORT_PROBS),
    })
    
    # Calculate Household Utilities
    household_df['Electricity_Consumption_KW'] = np.where(household_df['In_House_Solar_System'] == 1, 
                                                          household_df['Electricity_Consumption_KW'] * 0.6, 
                                                          household_df['Electricity_Consumption_KW'])
    household_df['Monthly_Energy_Consumption'] = household_df['Electricity_Consumption_KW'] + (household_df['Gas_Consumption_m3'] * 10.5)
    household_df['Carbon_Footprint_kg'] = (household_df['Electricity_Consumption_KW'] * 0.4) + (household_df['Gas_Consumption_m3'] * 2.0)
    
    household_df['Green_Fuel_Vehicle'] = np.where((household_df['Transportation_Type'] == 'Private Car') & (np.random.rand(total_households) > 0.7), 1, 0)
    household_df['Public_Transit_Green_Fuel'] = np.where(household_df['Transportation_Type'].isin(['LRT', 'Monorail', 'Electric Bus']), 1, 0)
    
    # 2. Expand to Individuals
    # np.repeat duplicates household data for each family member
    ind_df = household_df.loc[household_df.index.repeat(household_df['Family_Members'])].reset_index(drop=True)
    
    # Assign Role within family (0=Head, 1=Spouse, 2+=Children)
    ind_df['Family_Role_Index'] = ind_df.groupby('Real_Estate_ID').cumcount()
    
    total_individuals = len(ind_df)
    
    # 3. Vectorized Demographic Assignment based on Role
    # Head of Household (Husband usually)
    mask_head = ind_df['Family_Role_Index'] == 0
    ind_df.loc[mask_head, 'Gender'] = 'Male'
    ind_df.loc[mask_head, 'Age'] = np.random.randint(30, 65, mask_head.sum())
    ind_df.loc[mask_head, 'Marital_Status'] = 'Married'
    
    # Spouse (Wife usually, avg ~5-6 years younger)
    mask_spouse = ind_df['Family_Role_Index'] == 1
    ind_df.loc[mask_spouse, 'Gender'] = 'Female'
    ind_df.loc[mask_spouse, 'Age'] = np.random.randint(24, 60, mask_spouse.sum())
    ind_df.loc[mask_spouse, 'Marital_Status'] = 'Married'
    
    # Children/Dependents
    mask_child = ind_df['Family_Role_Index'] >= 2
    ind_df.loc[mask_child, 'Gender'] = np.random.choice(['Male', 'Female'], mask_child.sum())
    ind_df.loc[mask_child, 'Age'] = np.random.randint(0, 24, mask_child.sum())
    ind_df.loc[mask_child, 'Marital_Status'] = 'Single'
    
    ind_df['Year_of_Birth'] = year - ind_df['Age']
    
    # Predict Death Year (Life Expectancy ~ 74.5)
    life_expectancy = np.random.normal(74.5, 5, total_individuals)
    ind_df['Year_of_Death_Projected'] = ind_df['Year_of_Birth'] + life_expectancy
    ind_df['Year_of_Death_Projected'] = np.where(ind_df['Year_of_Death_Projected'] > year, ind_df['Year_of_Death_Projected'].astype(int), np.nan)
    
    # 4. Socio-Economic & Education
    ind_df['Education_Type'] = np.where((ind_df['Age'] >= 6) & (ind_df['Age'] <= 22), np.random.choice(EDU_TYPES, total_individuals), 'None')
    
    # Assign Grades
    conditions = [
        (ind_df['Age'] < 6),
        (ind_df['Age'] >= 6) & (ind_df['Age'] <= 18),
        (ind_df['Age'] > 18) & (ind_df['Age'] <= 22),
        (ind_df['Age'] > 22)
    ]
    choices = ['Pre-school', 'Grade 1-12', 'Undergraduate', 'Graduate/Professional']
    ind_df['Education_Grade'] = np.select(conditions, choices, default='Unknown')
    
    ind_df['University_Major'] = np.where(ind_df['Age'] > 18, np.random.choice(MAJORS, total_individuals), 'None')
    
    # Careers & Income
    careers = list(CAREERS.keys())
    ind_df['Career'] = np.where((ind_df['Age'] >= 22) & (ind_df['Age'] <= 64), np.random.choice(careers[:-1], total_individuals), 'Unemployed / Student / Child')
    
    # Assign random income within career bands
    def assign_income(career_series):
        incomes = np.zeros(len(career_series))
        for career, (min_inc, max_inc) in CAREERS.items():
            mask = career_series == career
            if mask.sum() > 0 and max_inc > 0:
                incomes[mask] = np.random.randint(min_inc, max_inc, mask.sum())
        return incomes
        
    ind_df['Income_Per_Year_EGP'] = assign_income(ind_df['Career'])
    ind_df['Telecommuting_Days_Week'] = np.where(ind_df['Career'].isin(['Software Engineer / AI Specialist', 'Data Scientist']), np.random.randint(2, 5, total_individuals), 0)
    
    # 5. Health Status
    ind_df['Health_Percent'] = np.where(ind_df['Age'] > 65, np.random.randint(40, 90, total_individuals), np.random.randint(75, 100, total_individuals))
    ind_df['Health_Condition'] = np.random.choice(HEALTH_CONDS, total_individuals, p=[0.05, 0.05, 0.02, 0.05, 0.03, 0.1, 0.05, 0.05, 0.6])
    
    bed_req_conditions = ['Cardiovascular Services', 'Oncology Services', 'Neurology']
    ind_df['Needs_Hospital_Bed'] = np.where(ind_df['Health_Condition'].isin(bed_req_conditions) & (np.random.rand(total_individuals) > 0.6), 1, 0)
    
    ind_df['Target_Pop_Density'] = TARGET_POP_DENSITY
    ind_df['Simulation_Year'] = year
    
    # Drop intermediate logic columns
    ind_df = ind_df.drop(columns=['Family_Role_Index'])
    
    return ind_df

# ==========================================
# 4. EXECUTION RUNNER
# ==========================================
if __name__ == "__main__":
    print(f"Starting Simulation: {PROJECT_NAME}")
    
    for year in range(START_YEAR, END_YEAR + 1):
        num_households = generate_households_for_year(year)
        
        # Generate the full year's data
        yearly_df = generate_yearly_data(year, num_households)
        
        # Chunking to keep files under 100 MB
        total_rows = len(yearly_df)
        num_chunks = int(np.ceil(total_rows / ROWS_PER_CHUNK))
        
        for i in range(num_chunks):
            start_idx = i * ROWS_PER_CHUNK
            end_idx = min((i + 1) * ROWS_PER_CHUNK, total_rows)
            chunk_df = yearly_df.iloc[start_idx:end_idx].copy()
            
            # Inject exactly 3% noise into the chunk
            chunk_df = inject_noise(chunk_df)
            
            file_name = f"NAC_Demographics_{year}_Part{i+1}.csv"
            file_path = os.path.join(OUTPUT_DIR, file_name)
            
            chunk_df.to_csv(file_path, index=False)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f" -> Saved {file_name} | Rows: {len(chunk_df):,} | Size: {file_size_mb:.2f} MB")
            
    print("Generation Complete. All demographic bounds and memory constraints successfully applied.")
