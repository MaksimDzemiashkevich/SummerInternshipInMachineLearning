from langchain.tools import tool

@tool
def convert_bar_to_mpa(value_bar: float) -> float:
    """Convert pressure from bar to megapascals (MPa). 1 bar = 0.1 MPa."""
    return round(value_bar * 0.1, 2)

@tool
def convert_psi_to_mpa(value_psi: float) -> float:
    """Convert pressure from psi to megapascals (MPa). 1 psi = 0.006895 MPa."""
    return round(value_psi * 0.006895, 2)

@tool
def convert_inch_to_mm(value_inch: float) -> float:
    """Convert length from inches to millimetres. 1 inch = 25.4 mm."""
    return round(value_inch * 25.4, 2)

@tool
def convert_lb_to_kg(value_lb: float) -> float:
    """Convert weight from pounds to kilograms. 1 lb = 0.4536 kg."""
    return round(value_lb * 0.4536, 2)

@tool
def convert_kelvin_to_celsius(value_k: float) -> float:
    """Convert temperature from Kelvin to Celsius."""
    return round(value_k - 273.15, 2)

@tool
def convert_fahrenheit_to_celsius(value_f: float) -> float:
    """Convert temperature from Fahrenheit to Celsius."""
    return round((value_f - 32) * 5 / 9, 2)

array_of_tools = [convert_bar_to_mpa, convert_psi_to_mpa, convert_inch_to_mm,
                  convert_lb_to_kg, convert_kelvin_to_celsius, convert_fahrenheit_to_celsius]