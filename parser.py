# Create and populate SQLite database from JSON files
# JSON files are downloaded from OBDgitscrape.py
# and stored in obd_car_files folder
import sqlite3
import json
import os
import re

# Create SQLite database
conn = sqlite3.connect('OBD.db')
cursor = conn.cursor()

# Create tables from schema.sql
with open(os.path.join(os.path.dirname(__file__), 'schema.sql'), "r", encoding="utf-8") as f:
    cursor.executescript(f.read())


def main(data, manufacturer, model, parameter_aliases):
    

    def classify_key_parameter(signal_id, signal_name):
    #Classify signals into key parameters based on signal_id and name
        signal_id_upper = signal_id.upper()
        name_upper = signal_name.upper()

        def record_alias(param):
            """Helper to store name_upper in alias list without duplicates."""
            if param in parameter_aliases and name_upper not in parameter_aliases[param]:
                parameter_aliases[param].append(name_upper)
            return param


        # Skip battery modules (long names with CMU patterns)
        if 'CMU' in signal_id_upper and ('VOLT' in signal_id_upper or 'TEMP' in signal_id_upper):
            return None
        
        # Check for HV Battery related parameters
        # if ('HVBAT' in signal_id_upper or 'HV_BAT' in signal_id_upper or 'HV BAT' in signal_id_upper):
        if 'MODULE' in name_upper or 'BMS' in signal_id_upper or 'SSB' in signal_id_upper or 'AUX' in signal_id_upper:
            return None # Skip HV Battery Module signals
        

        # Check for specific HV Battery SOC signals
        # Treat SOC and SOC_Display as same (differ by abit) vs SOC_RAW (different)
        if (signal_id_upper.endswith('HVBAT_SOC') 
            or signal_id_upper.endswith('HVBAT_SOC_DISP') 
            or signal_id_upper.endswith('HVBAT_SOC_DISPLAY') 
            or signal_id_upper.endswith('HVBAT_SOC_DISP_V2') 
            or signal_id_upper.endswith('HVBAT_RAW_SOC_DISP')
            or signal_id_upper.endswith('Q4ETRON_SOC')):  # Audi Q4 e-tron:
            record_alias('Pack_SOC')
            return 'Pack_SOC'
        
        elif signal_id_upper.endswith('_SOC') and (len(signal_id_upper.split('_')) <= 3):
            record_alias('Pack_SOC')
            return 'Pack_SOC'
        
        elif signal_id_upper.endswith('HVBAT_SOC_V2'):
            record_alias('Pack_SOC_V2')
            return 'Pack_SOC_V2'

        elif signal_id_upper.endswith('HVBAT_SOC_RAW') or signal_id_upper.endswith('HVBAT_RAW_SOC'):
            record_alias('Pack_SOC_RAW')
            return 'Pack_SOC_RAW'
        
        # elif signal_upper.endswith('HVBAT_SOC_DISPLAY') or signal_upper.endswith('HVBAT_SOC_DISPLAY_V2') or signal_upper.endswith('HVBAT_SOC_DISP'):
        # elif 'HVBAT_SOC_DISP' in signal_id_upper:
        #     return 'HVBAT_SOC_DISPLAY'
        
        # Check for HV Battery Pack Voltage and Current signals
        elif (signal_id_upper.endswith('HVBAT_VOLTAGE') 
                or signal_id_upper.endswith('HVBAT_V') 
                or signal_id_upper.endswith('HVBAT_VOLT') 
                or signal_id_upper.endswith('HVBAT_VOLTS')
                or signal_id_upper.endswith('HVBAT_VDC')
                or signal_id_upper.endswith('HVBAT_BUS_V')):
            record_alias('Pack_Voltage')
            return 'Pack_Voltage'
        
        elif signal_id_upper.endswith('HVBAT_CURRENT') or signal_id_upper.endswith('HVBAT_C') or signal_id_upper.endswith('HVBAT_CURR'):
            record_alias('Pack_Current')
            return 'Pack_Current'
    

        # SOH, SOH_V2, SOH_RAW
        elif signal_id_upper.endswith('HVBAT_SOH') or signal_id_upper.endswith('HVB_SOH'):
            record_alias('Pack_SOH')
            return 'Pack_SOH'   
        elif signal_id_upper.endswith('HVBAT_SOH_V2'):
            record_alias('Pack_SOH_V2')
            return 'Pack_SOH_V2'
        elif signal_id_upper.endswith('HVBAT_SOH_RAW'):
            record_alias('Pack_SOH_RAW')
            return 'Pack_SOH_RAW'
        
        elif ('VOLT' in signal_id_upper or 'VOLTAGE' in signal_id_upper) and 'CMU' not in signal_id_upper:
            return 'Voltage'
        elif ('CURR' in signal_id_upper or 'CURRENT' in signal_id_upper):
            return 'Current'
        
        # Motor Torque Signals
        if ('EM_TORQ' in signal_id_upper):
            # Extract RPM number if present
            import re
            rpm_match = re.search(r'TORQ_(\d*)', signal_id_upper)
            if rpm_match:
                record_alias('Motor_Torque')
                return 'Motor_' + 'Torque' + signal_id.split('_')[-1]  # Return RPM with number if present
            return 'Motor_Torque'
        
        

        # Check for Motor RPM
        if ('EM_RPM' in signal_id_upper or "MOTOR RPM" in name_upper):
            # Extract RPM number if present
            import re
            rpm_match = re.search(r'RPM_(\d*)', signal_id_upper) # returns if RPM_<number>
            if rpm_match:
                # <car model>_RPM_1 returns 'Motor_RPM_1', RPM returns 'Motor_RPM'
                record_alias('Motor_RPM')
                return 'Motor_' + signal_id.split('_')[-2] + signal_id.split('_')[-1]  
            return 'Motor_RPM'
        
        # elif 'RPM' in signal_upper:
        #     if len(signal_id.split('_')) > 2:
        #         return 'Motor_' + signal_id.split('_')[-2] + signal_id.split('_')[-1] 
        # return None
    

    def determine_ev_status(manufacturer, model):
    #Determine if vehicle is EV based on manufacturer and model
        if not manufacturer or not model or model.strip() == '':
            return None
        # model_upper = model.upper()
        # manufacturer_upper = manufacturer.upper()

        ev_models = [
        'Q4 e-tron', 'i3s', 'i3', 'i4', 'iX3',
        'Bolt EUV', 'Bolt EV', 'Equinox EV',
        'eC4 X', 'Mustang Mach',
        'IONIQ 5', 'IONIQ 6', 'IONIQ Electric', 'Kona Electric',
        'I PACE', 'EV6', 'EV9', 'Niro EV', 'Soul',
        'MG4', 'Cooper SE', 'Leaf', '2',
        'Macan Electric', 'Taycan',
        'bZ4X', 'Corsa e', 'e Golf'
        ]   

        # Check if model contains EV indicators
        if any(ev_model in model for ev_model in ev_models):
            return True
        
        # Check for electric indicators in model name
        if any(keyword in model for keyword in ['ELECTRIC', 'EV', 'BEV', 'E-', '_E']):
            return True
        
        # Default to False for known models, None for unknown
        return False
    


    def create_transmit_message(cmd_dict):
        #Create transmit message in format: 03 <command> <pad till 8 bytes>
        if "22" in cmd_dict:
            cmd_value = cmd_dict["22"]
            # Convert hex string to proper format and pad to 8 bytes
            message = f"03 22 {cmd_value[:2]} {cmd_value[2:]} 00 00 00 00"
            return message.strip()
        return "03 00 00 00 00 00 00 00"

    def create_command_string(cmd_dict):
        #Combine cmd_key and cmd_value into single command string
        if "22" in cmd_dict:
            return f"22{cmd_dict['22']}"
        return "00000000"
    
    # Returns dictionary as a string with values separated by ;
    def format_map(map_dict):
        return '; '.join(f"{k}={v['value']}" for k, v in map_dict.items())
    


    # Sort commands by command value for consistent ordering
    sorted_commands = sorted(data["commands"], key=lambda x: create_command_string(x["cmd"]))

    # Insert commands and signals
    for command in sorted_commands:
        # Collect signal information
        signal_ids = []
        signal_paths = []
        suggested_metrics = []
        
        for signal in command["signals"]:
            signal_ids.append(signal["id"])
            signal_paths.append(signal.get("path"))
            if "suggestedMetric" in signal:
                suggested_metrics.append(signal["suggestedMetric"])
        
        # Convert lists to comma-separated strings, making paths distinct
        signal_ids_str = ",".join(signal_ids)
        # Join unique, non-empty signal paths into a single comma-separated string
        # - str(p) converts all values to strings (avoids NoneType join errors)
        # - if p skips None or empty values
        # - dict.fromkeys(...) preserves insertion order while removing duplicates
        signal_paths_str = ",".join(        
            dict.fromkeys(str(p) for p in signal_paths if p)
            ) if signal_paths else ""

        suggested_metrics_str = ",".join(suggested_metrics) if suggested_metrics else None
        
        # Create command string and transmit message
        command_str = create_command_string(command["cmd"])
        transmit_message = create_transmit_message(command["cmd"])
    
        hdr = command["hdr"]
        hdr_hex = int(hdr, 16)  # Convert hex string to integer
        # Calculate response ID by adding 8 to the header value
        response_id = hex(hdr_hex + 8)[2:].upper()  # Convert back to hex string without '0x' prefix, and capslocked
        freq = command.get("freq")  
         

        cursor.execute('''
            INSERT INTO commands (
            manufacturer, model, request_id, response_id, command, transmit_message, frequency, signal_ids, signal_paths, suggested_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (manufacturer, model, command_str, command["hdr"], response_id, 
            transmit_message, command["freq"], signal_ids_str, signal_paths_str, suggested_metrics_str))
        
        command_id = cursor.lastrowid
         

        # Insert signals for this command
        for signal in command["signals"]:
            fmt = signal["fmt"]
            key_param = classify_key_parameter(signal["id"], signal["name"])

            # Check if model is EV
            type = determine_ev_status(manufacturer, model)
            if type:
                type = 'EV'
            else:
                type = ''

            bix = 0 if fmt.get("bix") == None else fmt.get("bix")
            end_bit = bix + (fmt["len"]-1)
            cursor.execute('''
                INSERT INTO signals (command_id, manufacturer, model, type, signal_id, name, request_id, response_id, command, transmit_message,  
                            frequency, map, start_bit, end_bit, 
                            len, min_value, max_value, mul, div, "add",
                            path, suggested_metric,
                            unit, key_parameter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (command_id, manufacturer, model, type, 
                signal["id"], signal["name"],
                # f'"{command["hdr"]}"', f'"{response_id}"',
                hdr, response_id,
                command_str, transmit_message,  
                # signal.get("description"), 
                freq, format_map(fmt.get("map") if fmt.get("map") else {}), 
                fmt.get("bix") if fmt.get("bix") else 0, 
                end_bit, 
                fmt["len"],
                fmt.get("min"), 
                fmt.get("max"), 
                fmt.get("mul"), 
                fmt.get("div"), 
                fmt.get("add"), signal.get("path"), signal.get("suggestedMetric"), fmt.get("unit"),
                key_param
                ))
            if key_param:
                # Insert into key_parameters table if it's a key parameter
                            cursor.execute('''
                INSERT INTO key_signals (command_id, manufacturer, model, type, signal_id, name, request_id, response_id, command, transmit_message,  
                            frequency, map, start_bit, end_bit, 
                            len, min_value, max_value, mul, div, "add",
                            path, suggested_metric,
                            unit, key_parameter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (command_id, manufacturer, model, type, 
                signal["id"], signal["name"],
                hdr, response_id,
                command_str, transmit_message,  
                freq, format_map(fmt.get("map") if fmt.get("map") else {}), 
                fmt.get("bix") if fmt.get("bix") else 0, 
                end_bit, 
                fmt["len"],
                fmt.get("min"), 
                fmt.get("max"), 
                fmt.get("mul"), 
                fmt.get("div"), 
                fmt.get("add"), signal.get("path"), signal.get("suggestedMetric"), fmt.get("unit"),
                key_param
                ))
    
    
def iterate_folder():

    # parameter aliases
    parameter_aliases = {
        'Pack_SOC': [],
        'Pack_SOC_V2': [],
        'Pack_SOC_RAW': [],
        'Pack_Voltage': [],
        'Pack_SOH': [],
        'Pack_SOH_V2': [],
        'Pack_Current': [],
        
        # 'Voltage': [],
        # 'Current': [],
        'Motor_Torque': [],
        'Motor_RPM': []
    }


    folder_path = os.path.join(os.path.dirname(__file__), 'obd_car_files')  # Adjust the path as needed
    files = os.listdir(folder_path)
    # file_list = [file.replace("_default.json", "") for file in files if file.endswith("_default.json")]
    files = [file for file in files if file.endswith("_default.json")]
    # Iterate over all files in the folder
    counter = 0
    for file in files:
        model = ""
        if counter <= 200:
            # print(f"Processing file: {file}")
            manufacturer = file.replace("_default.json", "").split("-")[0]
            try: 
                if len(file.replace("_default.json", "").split("-")) >= 2:
                    # Join all parts after the first hyphen to form the model name
                    model = (file.replace(manufacturer + "-", "").replace("_default.json", "")).replace("-", " ")
                else:
                    model = ""
                # else:
                #     model = file.replace("_default.json", "").split("-")[1]
            except:
                model = ""
            # print(f"Manufacturer: {manufacturer}, Model: {model}")
            counter+=1
            with open(os.path.join(folder_path, file), 'r') as f:
                data = json.load(f)

            # Parse the JSON data and store it in the SQLite database
            main(data, manufacturer, model, parameter_aliases)
    print(f"Processed {counter} files.")
    print(parameter_aliases)
    
def debug(manufacturer, model):    
    folder_path = r"C:/Users/Zu Kai/NUS/OBD_Explorer/obd_car_files"
    with open(os.path.join(folder_path, f"{manufacturer}-{model}_default.json"), 'r') as f: 
        data = json.load(f)
    main(data, manufacturer, model)

# debug("Ford", "Edge")
iterate_folder() 
conn.commit()
conn.close()

# def iterate_folder():
#     folder_path = r"C:/Users/Zu Kai/NUS/OBD_Explorer/obd_car_files"

#     # Iterate over all files in the folder
#     for i in range(2):
#         i+=1
#         for filename in os.listdir(folder_path):
#             if filename.endswith(".json"):
#                 # Read the JSON data from the file
#                 with open(os.path.join(folder_path, filename), 'r') as f:
#                     data = json.load(f)

#                 # Parse the JSON data and store it in the SQLite database
#                 main(data)

# # Read the JSON data from the uploaded file
# with open('uds_config.json', 'r') as f:
#     data = json.load(f)

# Retrieve manufacturer+model from the JSON data
# folder_path = r'C:/Users/Zu Kai/NUS/OBD_Explorer/obd_car_files'
# files = os.listdir(folder_path)
# file_list = [file.replace("_default.json", "") for file in files if file.endswith("_default.json")]
# print(file_list)


# i = 0

# for file in file_list:
#     if i < 1:
#         manufacturer = file.split("-")[0]
#         try: 
#             if len(file.split("-")) > 2:
#                 model = file.split("-")[1] + " " + file.split("-")[2]
#             else:
#                 model = file.split("-")[1]
#         except:
#             model = ""
#         print(f"Manufacturer: {manufacturer}, Model: {model}")
#     i+=1

# Parse the JSON data and store it in the SQLite database
# main(data)
# iterate_folder()
# conn.close()
# folder_path = r"C:/Users/Zu Kai/NUS/OBD_Explorer/obd_car_files"
# files = os.listdir(folder_path)
# print(files)