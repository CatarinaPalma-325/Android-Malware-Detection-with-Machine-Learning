import androguard.core.bytecodes.apk as apk
import numpy as np

from androguard.misc import AnalyzeAPK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from fuzzywuzzy import fuzz


def feature_extraction(apk_path, most_relevant_features):
    """
    Function to extract features from an APK file.

    Input:
        APK file path
        most relevant features

    Output:
        result of the mapping between the extracted features and the most relevant features
    """

    a, d, dx = AnalyzeAPK(apk_path)
    apk_file = apk.APK(apk_path)

    # Extract classes and methods
    classes = []
    methods = []
    for dex_file in apk_file.get_all_dex():
        dalvik_vm = DalvikVMFormat(dex_file)
        for dex_class in dalvik_vm.get_classes():
            classes.append(dex_class.get_name())
            for method in dex_class.get_methods():
                methods.append(method.get_name())

    # Extract activities
    activities = a.get_activities()

    # Extract intents from activities
    activities_intents = []
    for activity in activities:
        activity_intents = a.get_intent_filters("activity", activity)
        activities_intents.append(activity_intents)

    # Extract services
    services = a.get_services()

    # Extract intents from services
    services_intents = []
    for service in services:
        service_intents = a.get_intent_filters("service", service)
        services_intents.append(service_intents)

    # Extract receivers
    receivers = a.get_receivers()

    # Extract intents from receivers
    receivers_intents = []
    for receiver in receivers:
        receiver_intents = a.get_intent_filters("receiver", receiver)
        receivers_intents.append(receiver_intents)

    # Extract providers
    providers = a.get_providers()

    # Extract intents from providers
    providers_intents = []
    for provider in providers:
        provider_intents = a.get_intent_filters("provider", provider)
        providers_intents.append(provider_intents)

    intents = activities_intents + services_intents + receivers_intents + providers_intents

    all_intents_action_and_category = []
    for intent in intents:
        action = intent.get("action")
        if action is not None:
            for ac in action:
                all_intents_action_and_category.append(ac)
        category = intent.get("category")
        if category is not None:
            for c in category:
                all_intents_action_and_category.append(c)

    # Extract permissions
    permissions = a.get_permissions()

    # Extract hardware and software features
    hardware_software_features = a.get_features()

    # Aggregate all extracted app features
    app_extracted_features = permissions + hardware_software_features + activities + providers + receivers + services + all_intents_action_and_category + classes + methods

    # Initialize an empty lists to store matches
    matches = []
    extraction_result = []
    # Iterate through features and permissions to check for matches
    for required_feature in most_relevant_features:
        match_found = False  # Flag to track if a match is found for the current feature
        for app_feature in app_extracted_features:
            # Calculate the similarity score between the two strings
            similarity_score = fuzz.ratio(required_feature, app_feature)
            # Set a threshold for similarity
            threshold = 90
            # Check if the similarity score is above the threshold
            if similarity_score >= threshold or required_feature in app_feature:
                extraction_result.append(1)
                matches.append((required_feature, app_feature))
                match_found = True  # Set the flag to True if a match is found
                break  # Exit the inner loop if a match is found
        if not match_found:
            extraction_result.append(0)  # Append 0 if no match is found for the current feature

    print("\n---------- Matches found in feature mapping ----------\n")
    # Print the matching elements
    for match in matches:
        print(f"Match found: {match[0]} in {match[1]}")
    print("\n------------------------------------------------------\n")

    return np.array(extraction_result).reshape(1, -1)
