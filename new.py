import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from fpdf import FPDF
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from database import (
    add_equipment, delete_equipment, update_equipment, search_equipment,
    get_equipment_details, get_total_cost, get_equipment_data, get_student_data,
    get_sport_types, get_report_data, add_student, update_student, delete_student,
    add_sport, update_sport, delete_sport
)

# Database functions
def connect_db():
    conn = sqlite3.connect('sport_facility.db')
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def import_data(df, table_name):
    conn = connect_db()
    cursor = conn.cursor()
    
    # Get existing data from the table to check for duplicates
    existing_data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    
    # Ensure DataFrame columns match the table columns
    df.columns = existing_data.columns
    
    # Convert columns to the appropriate data types
    for col in df.columns:
        if col in existing_data.columns:
            if existing_data[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif existing_data[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif existing_data[col].dtype == 'datetime64[ns]':
                df[col] = pd.to_datetime(df[col], errors='coerce')
            else:
                df[col] = df[col].astype(str)
    
    # Find new data (rows in df that are not in existing_data)
    merged = pd.merge(df, existing_data, how='left', indicator=True)
    new_data = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    # Insert new data into the table
    if not new_data.empty:
        try:
            new_data.to_sql(table_name, conn, if_exists='append', index=False)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
    
    conn.close()


def get_sport_types():
    conn = connect_db()
    query = "SELECT * FROM sports"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def get_equipment_data():
    conn = connect_db()
    query = "SELECT * FROM equipment"
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def get_total_student_data():
    conn = connect_db()
    query = "SELECT * FROM students"
    data = pd.read_sql(query, conn)
    conn.close()
    return data


def get_student_data(start_date=None, end_date=None):
    conn = connect_db()
    query = 'SELECT * FROM students WHERE 1=1'
    params = []
    if start_date and end_date:
        query += ' AND (borrowing_date BETWEEN ? AND ? OR return_date BETWEEN ? AND ?)'
        params = [start_date, end_date, start_date, end_date]
    data = pd.read_sql(query, conn, params=params)
    conn.close()
    return data

def get_report_data(start_date=None, end_date=None):
    conn = connect_db()
    query = '''
    SELECT e.code AS equipment_code, e.name AS equipment_name, e.brand AS equipment_brand, e.status AS equipment_status,
           st.code AS student_code, st.name AS student_name, st.mobile_number AS student_no,
           st.borrowing_date, st.return_date,
           s.code AS sport_code, s.name AS sport_name
    FROM equipment e
    LEFT JOIN students st ON st.equipment_id = e.code
    LEFT JOIN sports s ON s.code = st.sports_code
    WHERE 1=1
    '''

    # Add date filtering if dates are provided
    if start_date and end_date:
        query += ' AND (st.borrowing_date BETWEEN ? AND ? OR st.return_date BETWEEN ? AND ?)'
        params = [start_date, end_date, start_date, end_date]
    else:
        params = []

    report_data = pd.read_sql(query, conn, params=params)
    conn.close()
    return report_data


def get_equipment_details(equipment_code):
    conn = sqlite3.connect('sport_facility.db')
    query = '''
    SELECT code, name, brand, status, purchase_date, cost
    FROM equipment
    WHERE code = ?
    '''
    equipment_details = pd.read_sql(query, conn, params=(equipment_code,))
    conn.close()
    return equipment_details


def get_total_cost(equipment_codes_list, start_date=None, end_date=None):
    conn = connect_db()
    placeholders = ', '.join('?' for _ in equipment_codes_list) if equipment_codes_list else ''
    query = '''
    SELECT *
    FROM equipment
    WHERE 1=1
    '''
    
    params = []
    
    if equipment_codes_list:
        query += f' AND code IN ({placeholders})'
        params.extend(equipment_codes_list)
    
    if start_date:
        start_date_str = start_date.strftime('%Y-%m-%d')
        query += ' AND purchase_date >= ?'
        params.append(start_date_str)
    
    if end_date:
        end_date_str = end_date.strftime('%Y-%m-%d')
        query += ' AND purchase_date <= ?'
        params.append(end_date_str)
    
    query += ' ORDER BY purchase_date'
    
    equipment_details = pd.read_sql_query(query, conn, params=params)
    total_cost = equipment_details['cost'].sum()
    conn.close()
    
    return equipment_details, total_cost


# sport 
def add_sport(code, name):
    conn = connect_db()
    query = "INSERT INTO sports (code, name) VALUES (?, ?)"
    try:
        conn.execute(query, (code, name))
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Sport with code {code} already exists.")
    finally:
        conn.close()

def delete_sport(code):
    conn = connect_db()
    query = "DELETE FROM sports WHERE code = ?"
    conn.execute(query, (code,))
    conn.commit()
    conn.close()

def update_sport(code, name):
    conn = connect_db()
    query = "UPDATE sports SET name = ? WHERE code = ?"
    conn.execute(query, (name, code))
    conn.commit()
    conn.close()

# equipments
def add_equipment(name, code, brand, bill_no, inward, outward, status, purchase_date, cost):
    conn = connect_db()
    query = "INSERT INTO equipment (name, code, brand, bill_no, inward, outward, status, purchase_date, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    try:
        conn.execute(query, (name, code, brand, bill_no, inward, outward, status, purchase_date, cost))
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Equipment with code {code} already exists.")
    finally:
        conn.close()

def delete_equipment(code):
    conn = connect_db()
    query = "DELETE FROM equipment WHERE code = ?"
    conn.execute(query, (code,))
    conn.commit()
    conn.close()

def update_equipment(code, name, brand, bill_no, inward, outward, status, purchase_date, cost):
    conn = connect_db()
    query = """
    UPDATE equipment
    SET name = ?, brand = ?, status = ?, bill_no = ?, inward = ?, outward = ?, purchase_date = ?, cost = ?
    WHERE code = ?
    """
    conn.execute(query, (name, brand, status, bill_no, inward, outward, purchase_date, cost, code))
    conn.commit()
    conn.close()

def search_equipment(status, code):
    conn = connect_db()
    query = "SELECT * FROM equipment WHERE 1=1"
    params = []
    if status != "All":
        query += " AND status = ?"
        params.append(status)
    if code:
        query += " AND code LIKE ?"
        params.append(f"%{code}%")
    
    data = pd.read_sql(query, conn, params=params)
    conn.close()
    return data

# General search function to query across multiple tables and fields
def search_database(search_query):
    conn = connect_db()
    tables = ['sports', 'equipment', 'students']
    result = pd.DataFrame()  # Initialize an empty DataFrame to store results

    for table in tables:
        query = f"SELECT * FROM {table} WHERE 1=1"
        params = []
        
        # Columns to search in each table (adjust as needed)
        if table == 'sports':
            searchable_columns = ['code', 'name']
        elif table == 'equipment':
            searchable_columns = ['code', 'name', 'brand', 'bill_no', "inward", "outward", "status"]
        elif table == 'students':
            searchable_columns = ['code', 'student_class', 'name', 'mobile_number' ]

        # Add conditions based on the search query
        conditions = []
        for column in searchable_columns:
            conditions.append(f"{column} LIKE ?")
            params.append(f"%{search_query}%")

        # Join conditions with OR
        if conditions:
            query += " AND (" + " OR ".join(conditions) + ")"
        else:
            continue  # Skip if no conditions are added
        
        # Debug print the query to check correctness
        print(f"Executing query: {query} with params: {params}")

        # Execute the query and fetch data
        try:
            table_result = pd.read_sql(query, conn, params=params)
            if not table_result.empty:
                table_result['source_table'] = table  # Add a column indicating which table the result is from
                result = pd.concat([result, table_result], ignore_index=True)
        except Exception as e:
            print(f"Error querying table {table}: {e}")

    conn.close()
    return result

# student
def add_student(code, student_class, name, equipment_id, sport_code, mobile_number, borrowing_date, return_date):
    conn = connect_db()
    query = '''
    INSERT INTO students (code, student_class, name, equipment_id, sports_code, mobile_number, borrowing_date, return_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    '''
    try:
        conn.execute(query, (code, student_class, name, equipment_id, sport_code, mobile_number, borrowing_date, return_date))
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError(f"Student with code {code} already exists.")
    finally:
        conn.close()

def delete_student(code):
    conn = connect_db()
    query = "DELETE FROM students WHERE code = ?"
    conn.execute(query, (code,))
    conn.commit()
    conn.close()

def update_student(student_code, new_student_class, new_student_name, new_equipment_id, new_sport_code, new_mobile_number, new_borrowing_date, new_return_date):
    conn = connect_db()
    query = '''
        UPDATE students
        SET student_class = ?, name = ?, equipment_id = ?, sports_code = ?, mobile_number = ?, borrowing_date = ?, return_date = ?
        WHERE code = ?
    '''
    try:
        conn.execute(query, (new_student_class, new_student_name, new_equipment_id, new_sport_code, new_mobile_number, new_borrowing_date, new_return_date, student_code))
        conn.commit()
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Error updating student. 'Equipment ID' or 'Student code' is not valid ! : {e}")
    finally:
        conn.close()

# PDF generation for dashboard
def generate_pdf(dataframe, title):
    """Generate a PDF from a DataFrame."""
    dataframe = dataframe.fillna('N/A')  # Handle missing values

    pdf = FPDF(orientation='L', unit='mm', format='A4')  # Set to landscape mode
    pdf.add_page()
    pdf.set_font("Arial", size=8)  # Reduced font size

    # Add title
    pdf.cell(270, 10, txt=title, ln=True, align="C")  # Adjusted width for landscape

    # Add table headers
    headers = list(dataframe.columns)
    col_width = 270 / len(headers)  # Divide available width by number of columns
    for header in headers:
        pdf.cell(col_width, 10, txt=header, border=1)
    pdf.ln()

    # Add table rows
    for i, row in dataframe.iterrows():
        for col in dataframe.columns:
            pdf.cell(col_width, 10, txt=str(row[col]), border=1)
        pdf.ln()

    return pdf


#pdf generator function for invetory and report pages
def invetory_generate_pdf(dataframe, title):
    """Generate a PDF from a DataFrame including a total cost row and index names."""
    
    # Create a copy of the dataframe and handle missing values
    dataframe = dataframe.fillna('N/A')
    
    # Initialize PDF
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.cell(270, 10, txt=title, ln=True, align="C")

    # Add table headers including index column
    headers = ['Index'] + list(dataframe.columns)
    col_width = 270 / len(headers)  # Divide available width by number of columns
    pdf.set_font("Arial", style='B', size=12)  # Bold for headers
    for header in headers:
        pdf.cell(col_width, 10, txt=header, border=1)
    pdf.ln()
    
    # Add table rows
    pdf.set_font("Arial", size=12)  # Regular font for data
    for idx, row in dataframe.iterrows():
        pdf.cell(col_width, 10, txt=str(idx), border=1)  # Index column
        for col in dataframe.columns:
            pdf.cell(col_width, 10, txt=str(row[col]), border=1)
        pdf.ln()

    # Output PDF as a byte stream
    pdf_output = pdf.output(dest='S').encode('latin1')
    
    return pdf_output

def safe_text(text):
    """Ensure text is safe for PDF generation."""
    if text is None:
        return ''
    return str(text).replace('\n', ' ').replace('\r', '')

#report page generate function 

def report_generate_pdf(dataframe, title):
    """Generate a PDF from a DataFrame."""
    dataframe = dataframe.fillna('N/A')  # Handle missing values

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=8)
    pdf.cell(280, 10, txt=title, ln=True, align="C")

    headers = list(dataframe.columns)
    col_width = 280 / len(headers)
    for header in headers:
        pdf.cell(col_width, 10, txt=header, border=1)
    pdf.ln()

    for i, row in dataframe.iterrows():
        for col in dataframe.columns:
            pdf.cell(col_width, 10, txt=safe_text(row[col]), border=1)
        pdf.ln()

    return pdf

def safe_text(text):
    """Ensure text is safe for PDF generation."""
    if text is None:
        return ''
    return str(text).replace('\n', ' ').replace('\r', '')



# dashboard_page 

def dashboard_page():
    st.title("Dashboard")


    st.subheader("Search")
        
    search_query = st.text_input("Search")

    if st.button("Search"):
        if search_query:
            result = search_database(search_query)
            if not result.empty:
                st.write(result)
            else:
                st.write("No results found.")
        else:
            st.write("Please enter a search query.")


    # Fetch data from database
    equipment_data = get_equipment_data()
    student_data = get_student_data()
    sport_data = get_sport_types()
    total_student_data = get_total_student_data()

    # Display student details
    with st.expander("Student Details", expanded=True):
        if student_data.empty:
            st.write("No student data available.")
        else:
            st.dataframe(student_data)

    # Display equipment details
    with st.expander("Equipment Details", expanded=True):
        if equipment_data.empty:
            st.write("No equipment data available.")
        else:
            st.dataframe(equipment_data)

    # Display sport details
    with st.expander("Sport Details", expanded=True):
        if sport_data.empty:
            st.write("No sport data available.")
        else:
            st.dataframe(sport_data)


    # Display grid 
    col1, col2 = st.columns(2)
    bottom_left_column, bottom_right_column = st.columns(2)


    # Display total counts
    with col1:
        st.metric(
            label="Total Available Equipment", 
            value=len(equipment_data)
            )

        

    with col2:
        st.metric(
            label="Total Students Who Borrowed Equipment", 
            value=len(student_data)
            )
        
    

    with bottom_right_column:
    # Plotly bar chart
        with st.expander("Total_bar_chart", expanded=True):
            x_data = ['Total_Equipments', 'Total_Students', 'Total_Sports']
            y_data = [len(equipment_data), len(total_student_data), len(sport_data)]

            fig = go.Figure([go.Bar(x=x_data, y=y_data, text=y_data, textposition="auto")])
            fig.update_layout(
                # Remove fixed width and height to allow responsiveness
                margin=dict(l=0, r=0, t=0, b=0)  # Optional: adjust margins if needed
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with bottom_left_column:
        # Pie chart of active/inactive equipment
        with st.expander("Active vs Inactive Equipment", expanded=True):
            if not equipment_data.empty:
                active_equipment = equipment_data[equipment_data['status'] == 'Active']
                inactive_equipment = equipment_data[equipment_data['status'] == 'Inactive']

                sizes = [len(active_equipment), len(inactive_equipment)]
                labels = 'Active', 'Inactive'
                
                fig1, ax1 = plt.subplots()
                try:
                    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    st.pyplot(fig1)
                except ValueError as e:
                    st.error(f"Error creating pie chart: {e}")
            else:
                st.write("No equipment data available for the pie chart.")
    
    # Display and generate report PDF
    with st.expander("Generate Report", expanded=True):
        if st.button("Generate PDF Report"):
            report_data = get_report_data()
            if not report_data.empty:
                pdf = generate_pdf(report_data, "Sport Facility Report")
                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="Download PDF",
                    data=pdf_output,
                    file_name="Sport_Facility_Report.pdf",
                    mime="application/pdf"
                )
            else:
                st.write("No report data available.")


    st.subheader("Upload 'CSV' or 'PDF'")
    st.subheader("""Importing data into your SQLite database from "Excel files" or "CSV",
                 You should ensure that the Excel files match the expected schema of your database tables. Below are the example formats for the Excel files, including the required column names and a brief description of the content:

                1.sports: code, name
                2.equipment: name, code, brand, bill_no, inward, outward, status, purchase_date (e.g., "2023-01-01"), cost
                3.students: student_code, student_class, student_name, equipment_id, sport_code, mobile_number, borrowing_date (e.g., "2023-01-10"), return_date (e.g., "2023-02-10")
                After preparing your data files, follow these steps to import the data:

                1.Select the table to import data into, corresponding to the data file you have uploaded.
                2.Click on "Import".
                Your data will be successfully imported into the selected table.
                 """)
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])

    if uploaded_file:
        file_type = uploaded_file.type
        if file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file, sheet_name=None)  # Read all sheets
            sheet_names = list(df.keys())
            sheet_name = st.selectbox("Select sheet", sheet_names)
            df = df[sheet_name]
        
        st.write("Data preview:")
        st.write(df.head())

        table_name = st.selectbox("Select table to import data", ["sports", "equipment", "students"])

        if st.button("Import Data"):
            try:
                import_data(df, table_name)
                st.success(f"Data imported successfully to the {table_name} table!")
            except Exception as e:
                st.error(f"Error importing data: {e}")

# facility_management page

def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = "Add Sport"

def facility_management_page():
    initialize_session_state()
    
    st.title("Sport Facility Management")

    # Button-based navigation
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Add Sport", key="add_sport_nav"):
            st.session_state.page = "Add Sport"
    with col2:
        if st.button("Manage Sports", key="manage_sports_nav"):
            st.session_state.page = "Manage Sports"
    with col3:
        if st.button("Add Student", key="add_student_nav"):
            st.session_state.page = "Add Student"
    with col4:
        if st.button("Manage Students", key="manage_students_nav"):
            st.session_state.page = "Manage Students"

    # Display content based on selected page
    if st.session_state.page == "Add Sport":
        st.subheader("Add Sport")
        sport_code = st.text_input("Sport Code")
        sport_name = st.text_input("Sport Name")
        
        if st.button("Add Sport", key="add_sport_now"):
            add_sport(sport_code, sport_name)
            st.success("Sport added successfully!")

    elif st.session_state.page == "Manage Sports":
        st.subheader("Manage Sports")
        sport_code = st.text_input("Sport Code (for update/delete)")
        
        # Delete Sport
        if st.button("Delete Sport", key="delete_sport"):
            delete_sport(sport_code)
            st.success("Sport deleted successfully!")
        
        # Update Sport
        new_sport_name = st.text_input("New Sport Name")
        if st.button("Update Sport", key="update_sport"):
            update_sport(sport_code, new_sport_name)
            st.success("Sport updated successfully!")

    elif st.session_state.page == "Add Student":
        st.subheader("Add Student")
        student_code = st.text_input("Student ID")
        student_class = st.text_input("Student Class")
        student_name = st.text_input("Student Name")
        equipment_id = st.text_input("Equipment ID")
        sport_code = st.text_input("Sports Code")
        mobile_number = st.text_input("Mobile Number")
        borrowing_date = st.date_input("Equipment Borrowing Date")
        return_date = st.date_input("Equipment Return Date")
        
        if st.button("Add Student", key="add_student_now"):
            try:
                add_student(student_code, student_class, student_name, equipment_id, sport_code, mobile_number, borrowing_date, return_date)
                st.success("Student added successfully!")
            except ValueError as e:
                st.error(e)

    elif st.session_state.page == "Manage Students":
        st.subheader("Manage Students")
        student_code = st.text_input("Student ID (for update/delete)")
        
        # Delete Student
        if st.button("Delete Student", key="delete_student"):
            delete_student(student_code)
            st.success("Student deleted successfully!")
        
        # Update Student
        new_student_name = st.text_input("New Student Name")
        new_equipment_id = st.text_input("New Equipment ID")
        new_student_class = st.text_input("New Student Class")
        new_sport_code = st.text_input("Sports Code")
        new_mobile_number = st.text_input("New Mobile Number")
        new_borrowing_date = st.date_input("New Equipment Borrowing Date")
        new_return_date = st.date_input("New Equipment Return Date")

        if st.button("Update Student", key="update_student"):
            try:
                update_student(student_code, new_student_class, new_student_name, new_equipment_id, new_sport_code, new_mobile_number, new_borrowing_date, new_return_date)
                st.success("Student updated successfully!")
            except ValueError as e:
                st.error(e)


# inventory_mangement page
def inventory_management_page():
    initialize_session_state()
    
    st.title("Sport Equipment Inventory Management")

    # Button-based navigation
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Add Equipment", key="add_equipment_nav"):
            st.session_state.page = "Add Equipment"
    with col2:
        if st.button("Search Equipment", key="search_equipment_nav"):
            st.session_state.page = "Search Equipment"
    with col3:
        if st.button("Manage Equipment", key="manage_equipment_nav"):
            st.session_state.page = "Manage Equipment"
    with col4:
        if st.button("Equipment Details and Cost", key="details_cost_nav"):
            st.session_state.page = "Equipment Details and Cost"

    # Display content based on selected page
    if st.session_state.page == "Add Equipment":
        st.subheader("Add Equipment")
        name = st.text_input("Equipment Name")
        code = st.text_input("Equipment Code")
        brand = st.text_input("Brand")
        bill_no = st.text_input("Bill Number")
        inward = st.text_input("Inward")
        outward = st.text_input("Outward")
        status = st.selectbox("Status", ["Active", "Inactive"], key="status")
        purchase_date = st.date_input("Date of Purchase", key="purchase_date")
        cost = st.number_input("Cost", min_value=0.0, key="cost")
        
        if st.button("Add Equipment", key="add_equipment_now"):
            add_equipment(name, code, brand, bill_no, inward, outward, status, purchase_date, cost)
            st.success("Equipment added successfully!")

    elif st.session_state.page == "Search Equipment":
        st.subheader("Search Equipment")
        status = st.selectbox("Status", ["All", "Active", "Inactive"], key="search_status")
        code = st.text_input("Equipment Code", key="search_code")
        
        if st.button("Search", key="search_now"):
            result = search_equipment(status, code)
            st.write(result)

    elif st.session_state.page == "Manage Equipment":
        st.subheader("Manage Equipment")
        equipment_code = st.text_input("Equipment Code (for update/delete)", key="manage_code")

        # Delete Equipment
        if st.button("Delete Equipment", key="delete_equipment"):
            delete_equipment(equipment_code)
            st.success("Equipment deleted successfully!")

        # Update Equipment
        new_name = st.text_input("New Equipment Name", key="new_name")
        new_brand = st.text_input("New Brand", key="new_brand")
        new_status = st.selectbox("New Status", ["Active", "Inactive"], key="new_status")
        new_bill_no = st.text_input("New Bill Number", key="new_bill_no")
        new_inward = st.text_input("New Inward", key="new_inward")
        new_outward = st.text_input("New Outward", key="new_outward")
        new_purchase_date = st.date_input("New Date of Purchase", key="new_purchase_date")
        new_cost = st.number_input("New Cost", min_value=0.0, key="new_cost")
        if st.button("Update Equipment", key="update_equipment"):
            update_equipment(equipment_code, new_name, new_brand, new_bill_no, new_inward, new_outward, new_status, new_purchase_date, new_cost)
            st.success("Equipment updated successfully!")

    elif st.session_state.page == "Equipment Details and Cost":
        st.subheader("Equipment Details and Cost")
        
        # Get Equipment Details
        equipment_code = st.text_input("Enter Equipment Code for Details:", key="details_code")
        if st.button("Get Equipment Details", key="get_details"):
            if equipment_code:
                equipment_details = get_equipment_details(equipment_code)
                st.write("Equipment Details:")
                st.dataframe(equipment_details)
            else:
                st.warning("Please enter a valid equipment code.")
        
        # Calculate Total Cost
        selected_equipment_codes = st.text_area("Enter Equipment Codes (comma-separated) for Total Cost:", key="cost_codes")
        if st.button("Calculate Total Cost", key="calculate_cost"):
            if selected_equipment_codes:
                equipment_codes_list = [code.strip() for code in selected_equipment_codes.split(',')]
                equipment_details, total_cost = get_total_cost(equipment_codes_list)
                
                # Add a summary row to the DataFrame and set the index name
                summary_row = pd.DataFrame([[""] * (len(equipment_details.columns) - 1) + [total_cost]],
                                           columns=equipment_details.columns, index=["Total"])
                equipment_details = pd.concat([equipment_details, summary_row], ignore_index=False)
                
                # Set the index name
                equipment_details.index.name = 'Index'
                
                st.session_state['dataframe'] = equipment_details
                st.write("Selected Equipment Details:")
                st.dataframe(equipment_details)
                
                # Generate PDF
                pdf_output = invetory_generate_pdf(equipment_details, "Equipment Details and Total Cost")
                
                # Save PDF to a BytesIO buffer
                buf = BytesIO()
                buf.write(pdf_output)  # Write the byte content
                buf.seek(0)
                
                # Provide download link
                st.download_button(
                    label="Download PDF",
                    data=buf.getvalue(),  # Use getvalue() to get the byte content
                    file_name="equipment_details_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Please enter valid equipment codes.")

        # Calculate Total Cost by Date Range
        st.subheader("Calculate Total Cost by Date Range")
        start_date = st.date_input("Start Date", key="start_date")
        end_date = st.date_input("End Date", key="end_date")

        if st.button("Calculate Total Cost by Date", key="calculate_by_date"):
            selected_equipment_codes = st.text_input("Enter Equipment Codes (comma-separated)", key="codes_date")
            equipment_codes_list = [code.strip() for code in selected_equipment_codes.split(',')] if selected_equipment_codes else []
    
            equipment_details, total_cost = get_total_cost(equipment_codes_list, start_date=start_date, end_date=end_date)
    
            summary_row = pd.DataFrame([[""] * (len(equipment_details.columns) - 1) + [total_cost]],
                                columns=equipment_details.columns, index=["Total"])
            equipment_details = pd.concat([equipment_details, summary_row], ignore_index=False)
    
            equipment_details.index.name = 'Index'
    
            st.session_state['dataframe'] = equipment_details
            st.write("Filtered Equipment Details:")
            st.dataframe(equipment_details)
    
            # Generate PDF
            pdf_output = invetory_generate_pdf(equipment_details, "Filtered Equipment Details and Total Cost")
    
            buf = BytesIO()
            buf.write(pdf_output)
            buf.seek(0)
    
            st.download_button(
                label="Download PDF",
                data=buf.getvalue(),
                file_name="filtered_equipment_details_report.pdf",
                mime="application/pdf"
            )

# report page
def report_page():
    st.title("Reports")

    report_option = st.selectbox(
        "Choose report type", 
        ["All Data", "Equipment", "Sports", "Students"]
    )

    # Date Range Pickers
    st.subheader("Select Date Range")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    if start_date > end_date:
        st.error("Start date must be before end date.")

    if report_option == "All Data":
        st.subheader("Combined Report")
        report_data = get_report_data(start_date, end_date)
        
        with st.expander("View Combined Report Data", expanded=True):
            if not report_data.empty:
                st.write(report_data)
                if st.button("Download Combined PDF Report"):
                    pdf = report_generate_pdf(report_data, "Combined Sport Facility Report")
                    pdf_output = pdf.output(dest='S').encode('latin1')
                    st.download_button(
                        label="Download Combined PDF",
                        data=pdf_output,
                        file_name="Combined_Sport_Facility_Report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.write("No data available.")

    elif report_option == "Equipment":
        st.subheader("Equipment Report")
        equipment_data = get_equipment_data()
        if not equipment_data.empty:
            # Convert date column to datetime
            equipment_data['purchase_date'] = pd.to_datetime(equipment_data['purchase_date'], errors='coerce')

            # Filter data based on date range
            filtered_data = equipment_data[
                (equipment_data['purchase_date'] >= pd.Timestamp(start_date)) & 
                (equipment_data['purchase_date'] <= pd.Timestamp(end_date))
            ]
            with st.expander("View Equipment Data"):
                st.write(filtered_data)
            if st.button("Download Equipment PDF Report"):
                pdf = report_generate_pdf(filtered_data, "Equipment Report")
                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="Download Equipment PDF",
                    data=pdf_output,
                    file_name="Equipment_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.write("No equipment data available.")

    elif report_option == "Sports":
        st.subheader("Sports Report")
        sport_types = get_sport_types()
        if not sport_types.empty:
            with st.expander("View Sports Data"):
                st.write(sport_types)
            if st.button("Download Sports PDF Report"):
                pdf = report_generate_pdf(sport_types, "Sports Report")
                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="Download Sports PDF",
                    data=pdf_output,
                    file_name="Sports_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.write("No sports data available.")

    elif report_option == "Students":
        st.subheader("Students Report")
        student_data = get_student_data(start_date, end_date)
        if not student_data.empty:
            # Convert date columns to datetime
            student_data['borrowing_date'] = pd.to_datetime(student_data['borrowing_date'], errors='coerce')
            student_data['return_date'] = pd.to_datetime(student_data['return_date'], errors='coerce')

            # Filter data based on date range
            filtered_data = student_data[
                (student_data['borrowing_date'] >= pd.Timestamp(start_date)) & 
                (student_data['return_date'] <= pd.Timestamp(end_date))
            ]
            with st.expander("View Students Data"):
                st.write(filtered_data)
            if st.button("Download Students PDF Report"):
                pdf = report_generate_pdf(filtered_data, "Students Report")
                pdf_output = pdf.output(dest='S').encode('latin1')
                st.download_button(
                    label="Download Students PDF",
                    data=pdf_output,
                    file_name="Students_Report.pdf",
                    mime="application/pdf"
                )
        else:
            st.write("No student data available.")

    # Collapsible section for data management
    with st.expander("Manage Data", expanded=True):
        # Manage Equipment
        st.subheader("Manage Equipment")
        with st.form("Equipment Form"):
            col1, col2 = st.columns(2)
            with col1:
                eq_code = st.text_input("Equipment Code")
                eq_name = st.text_input("Equipment Name")
                eq_brand = st.text_input("Brand")
                eq_bill_no = st.text_input("Bill Number")
                eq_inward = st.text_input("Inward")
                eq_outward = st.text_input("Outward")
            with col2:
                eq_status = st.selectbox("Status", ["Active", "Inactive"])
                eq_purchase_date = st.date_input("Date of Purchase")
                eq_cost = st.number_input("Cost", min_value=0.0)

            action = st.selectbox("Select Action", ["Add", "Update", "Delete"])
            submitted = st.form_submit_button("Submit")
            if submitted:
                if action == "Add":
                    add_equipment(eq_name, eq_code, eq_brand, eq_bill_no, eq_inward, eq_outward, eq_status, eq_purchase_date, eq_cost)
                    st.success("Equipment added successfully!")
                elif action == "Update":
                    update_equipment(eq_code, eq_name, eq_brand, eq_bill_no, eq_inward, eq_outward, eq_status, eq_purchase_date, eq_cost)
                    st.success("Equipment updated successfully!")
                elif action == "Delete":
                    delete_equipment(eq_code)
                    st.success("Equipment deleted successfully!")

        # Manage Students
        st.subheader("Manage Students")
        with st.form("Student Form"):
            col1, col2 = st.columns(2)
            with col1:
                student_code = st.text_input("Student ID")
                students_class = st.text_input("Student Class")
                student_name = st.text_input("Student Name")
                equipment_id = st.text_input("Equipment ID")
                sport_code = st.text_input("Sports Code")
                mobile_number = st.text_input("Mobile Number")
            with col2:
                borrowing_date = st.date_input("Borrowing Date")
                return_date = st.date_input("Return Date")

            action = st.selectbox("Select Action", ["Add", "Update", "Delete"])
            submitted = st.form_submit_button("Submit")
            if submitted:
                if action == "Add":
                    add_student(student_code, students_class, student_name, equipment_id, sport_code, mobile_number, borrowing_date, return_date)
                    st.success("Student added successfully!")
                elif action == "Update":
                    update_student(student_code, students_class, student_name, equipment_id, sport_code, mobile_number, borrowing_date, return_date)
                    st.success("Student updated successfully!")
                elif action == "Delete":
                    delete_student(student_code)
                    st.success("Student deleted successfully!")

        # Manage Sports
        st.subheader("Manage Sports")
        with st.form("Sport Form"):
            sport_code = st.text_input("Sport Code")
            sport_name = st.text_input("Sport Name")

            action = st.selectbox("Select Action", ["Add", "Update", "Delete"])
            submitted = st.form_submit_button("Submit")
            if submitted:
                if action == "Add":
                    add_sport(sport_code, sport_name)
                    st.success("Sport added successfully!")
                elif action == "Update":
                    update_sport(sport_code, sport_name)
                    st.success("Sport updated successfully!")
                elif action == "Delete":
                    delete_sport(sport_code)
                    st.success("Sport deleted successfully!")


# Dictionary to store user credentials
# For demonstration purposes, using a dictionary. In production, use a database and secure password storage.
# Initialize the user dictionary in session state if it doesn't exist
# Initialize the user dictionary in session state if it doesn't exist
if 'users' not in st.session_state:
    st.session_state['users'] = {
        "admin": "admin"  # Default admin credentials
    }

# Page functions
def login_page():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type='password')

    if st.button("Login"):
        # Check if the entered credentials match any stored user
        if username in st.session_state['users'] and st.session_state['users'][username] == password:
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
            st.experimental_rerun()  # Refresh the page to show the app content
        else:
            st.error("Invalid username or password")

def register_page():
    st.title("Register")

    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')

    if st.button("Register"):
        if new_password != confirm_password:
            st.error("Passwords do not match!")
        elif new_username in st.session_state['users']:
            st.error("Username already exists!")
        else:
            st.session_state['users'][new_username] = new_password
            st.success("User registered successfully!")

def manage_users_page():
    st.title("Manage Users")

    # Display current users
    st.subheader("Current Users")
    for username in st.session_state['users']:
        st.write(username)

    # Delete user
    st.subheader("Delete User")
    delete_username = st.text_input("Username to delete")
    if st.button("Delete User"):
        if delete_username in st.session_state['users']:
            del st.session_state['users'][delete_username]
            st.success(f"User '{delete_username}' deleted successfully!")
        else:
            st.error("Username not found!")

    # Update user
    st.subheader("Update User Password")
    update_username = st.text_input("Username to update")
    new_password = st.text_input("New Password", type='password')
    confirm_new_password = st.text_input("Confirm New Password", type='password')

    if st.button("Update User"):
        if update_username not in st.session_state['users']:
            st.error("Username not found!")
        elif new_password != confirm_new_password:
            st.error("Passwords do not match!")
        else:
            st.session_state['users'][update_username] = new_password
            st.success(f"Password for user '{update_username}' updated successfully!")


# loading page main.app
def main():
    st.set_page_config(page_title="Sports Facilities Management", page_icon=":sports_medal:",layout="wide")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if st.session_state['logged_in']:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select a page", ["Dashboard",  "Inventory Management", "Facility Management", "Reports", "Manage Users", "Logout"])

        if page == "Dashboard":
            dashboard_page()
        elif page == "Inventory Management":
            inventory_management_page()
        elif page == "Facility Management":
            facility_management_page()
        elif page == "Reports":
            report_page()
        elif page == "Manage Users":
            manage_users_page()
        elif page == "Logout":
            st.session_state['logged_in'] = False
            st.session_state['username'] = None  # Clear logged in username
            st.experimental_rerun()
    else:
        page = st.sidebar.radio("Login or Register", ["Login", "Register"])

        if page == "Login":
            login_page()
        elif page == "Register":
            register_page()

if __name__ == "__main__":
    main()
