from django.shortcuts import render
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from .utils import predict_prices
import pandas as pd
import os
import numpy as np
# Create your views here.
def login(request):

        return render(request, 'login.html')

def Analysis(request):

        return render(request,'Analysis.html')

def prediction(request):

        return render(request,'prediction.html')
def convert_to_float(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


from datetime import datetime, timedelta

def get_weekday_name(index):
    """Return the name of the weekday given an index."""
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return weekdays[index % 7]

def prediction_view(request):
    if request.method == 'POST':
        supermarket = request.POST.get('supermarket')
        product_name = request.POST.get('product_name')
        num_days = int(request.POST.get('day')) 
        start_weekday = request.POST.get('weekday')  # Starting weekday

        try:
            # Perform predictions
            sequence_length = 10
            predictions, dynamic_prices = predict_prices(supermarket, product_name, sequence_length, num_days, start_weekday)

            # Convert predictions and dynamic_prices to JSON serializable format
            predictions = [convert_to_float(p) for p in predictions]
            dynamic_prices = [convert_to_float(dp) for dp in dynamic_prices]

            # Calculate the sequence of weekdays in reverse
            weekdays_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            start_index = weekdays_list.index(start_weekday)
            
            # Generate the weekdays list in reverse
            weekdays = [get_weekday_name((start_index - i) % 7) for i in range(num_days)]

            # Zip the data for easier iteration in the template
            results = zip(
                range(1, num_days + 1),
                [product_name] * num_days,
                weekdays[::-1],  # Reverse the list to match the day sequence
                [supermarket] * num_days,
                predictions,
                dynamic_prices
            )

            # Pass the data to the template
            context = {
                'results': results,
                'product_name': product_name,
                'weekday': start_weekday,
                'supermarket': supermarket,
            }
            return render(request, 'prediction_result.html', context)

        except ValueError as e:
            return render(request, 'prediction.html', {'error': str(e)})
    else:
        return HttpResponse(status=405)




    
# Define a mapping from weekday names to numerical values
WEEKDAY_MAP = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

def competitor_prices_view(request):
    if request.method == 'POST':
        # Get form data
        supermarket = request.POST.get('supermarket')
        product_name = request.POST.get('product_name')
        day = int(request.POST.get('day', 0))  # Convert day to integer with default 0
        weekday_name = request.POST.get('weekday')

        # Convert weekday name to numerical value
        weekday = WEEKDAY_MAP.get(weekday_name, -1)  # Default to -1 if not found
        
        # Check if the weekday is valid
        if weekday == -1:
            return render(request, 'compititor_prices_view.html', {'error': 'Invalid weekday provided'})
        
        # Load dataset
        try:
            df = pd.read_csv('myapp/dataset/Demand_Dataset1.csv')
        except FileNotFoundError:
            return render(request, 'compititor_prices_view.html', {'error': 'Dataset file not found'})

        # Print for debugging
        print("Supermarket:", supermarket)
        print("Product Name:", product_name)
        print("Day:", day)
        print("Weekday:", weekday)

        # Print DataFrame columns and a few rows for debugging
        print("DataFrame Columns:", df.columns)
        print("DataFrame Head:", df.head())
        
        # Filter dataset for competitor supermarkets
        df_filtered = df[(df['supermarket'] != supermarket) & 
                         (df['product'] == product_name) & 
                         (df['weekday'] == weekday) &
                         (df['month'] == 1) &
                         (df['year'] == 2024)]
        
        # Print filtered DataFrame for debugging
        print("Filtered DataFrame:", df_filtered)
        
        # Extract relevant information
        competitor_prices = df_filtered[['supermarket', 'unit_price']]

        # Print competitor prices for debugging
        print("Competitor Prices:", competitor_prices)
        
        return render(request, 'compititor_prices_view.html', {'competitor_prices': competitor_prices})
    
    return render(request, 'compititor_prices_view.html')