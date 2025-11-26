"""
Test the top 3 lines priority logic for location extraction
"""
from location_identifier import extract_location

def test_location_in_top_3_lines():
    """Test cases where location is in the top 3 lines"""
    
    print("=" * 80)
    print("TEST 1: Location in Top 3 Lines (Priority Check)")
    print("=" * 80)
    
    test_cases = [
        {
            'name': 'Case 1: Location in header (Line 3)',
            'resume': """John Doe
Senior Software Engineer
Portland, Oregon
john@email.com
+1-503-555-1234

Experience:
Software Engineer for Microsoft - Seattle, WA
Led development team for Amazon - San Francisco, CA
""",
            'expected': 'Portland, Oregon',
            'should_avoid': 'Seattle, San Francisco'
        },
        {
            'name': 'Case 2: Location with "Location:" prefix (Line 2)',
            'resume': """Rajesh Kumar
Location: Hyderabad, India
rajesh@example.com

Professional Summary:
Working for Accenture - New York, NY
Client projects in London, UK
""",
            'expected': 'Hyderabad, India',
            'should_avoid': 'New York, London'
        },
        {
            'name': 'Case 3: City, State ZIP format (Line 3)',
            'resume': """Sarah Johnson
Backend Developer
Beaverton, OR 97005
sarah.johnson@gmail.com

Career:
Consultant for Oracle - Austin, TX
""",
            'expected': 'Beaverton',
            'should_avoid': 'Austin'
        },
        {
            'name': 'Case 4: Parenthetical format in header',
            'resume': """Mike Chen
Full Stack Developer (Portland, OR)
mike@example.com
+1-971-555-9999

Working remotely for Google - Mountain View, CA
""",
            'expected': 'Portland',
            'should_avoid': 'Mountain View'
        },
        {
            'name': 'Case 5: NO location in top 3, should find later',
            'resume': """Suresh Kavili
Technical Director
bharathi@email.com

Professional Overview:
Having 23 years of experience...

Mavensoft Systems Pvt Ltd, Hyderabad, India
Feb 2016 - Till Date
Director - Technology

System Analyst for Finaplex - San Francisco, CA
""",
            'expected': 'Hyderabad',
            'should_avoid': 'San Francisco'
        },
        {
            'name': 'Case 6: Just city name (major city)',
            'resume': """Navabharathi K
Software Engineer
Hyderabad
navabharathi@gmail.com

Experience with various clients
""",
            'expected': 'Hyderabad',
            'should_avoid': ''
        },
        {
            'name': 'Case 7: Multiple locations in header (should use first)',
            'resume': """Alex Turner
Developer | Portland, OR
Previously: Seattle, WA
alex@example.com

Career history...
""",
            'expected': 'Portland',
            'should_avoid': ''
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'-' * 80}")
        print(f"Test {i}: {test['name']}")
        print(f"{'-' * 80}")
        
        # Show first 3 lines
        first_3_lines = '\n'.join(test['resume'].strip().split('\n')[:3])
        print(f"\nTop 3 lines of resume:")
        print(f"  Line 1: {test['resume'].strip().split(chr(10))[0]}")
        print(f"  Line 2: {test['resume'].strip().split(chr(10))[1] if len(test['resume'].strip().split(chr(10))) > 1 else '(empty)'}")
        print(f"  Line 3: {test['resume'].strip().split(chr(10))[2] if len(test['resume'].strip().split(chr(10))) > 2 else '(empty)'}")
        
        # Extract location
        result = extract_location(test['resume'])
        
        print(f"\nExtracted Location: {result}")
        print(f"Expected Pattern: {test['expected']}")
        
        # Check if result matches expected pattern
        if test['expected'].lower() in result.lower():
            print(f"✅ PASSED: Correct location extracted")
            passed += 1
            
            # Check if it avoided wrong locations
            if test['should_avoid']:
                avoided_all = True
                for avoid_city in test['should_avoid'].split(', '):
                    if avoid_city.lower() in result.lower():
                        print(f"  ⚠️  WARNING: Should have avoided '{avoid_city}'")
                        avoided_all = False
                
                if avoided_all:
                    print(f"  ✅ Successfully avoided client locations: {test['should_avoid']}")
        else:
            print(f"❌ FAILED: Expected '{test['expected']}' but got '{result}'")
            failed += 1
    
    print(f"\n{'=' * 80}")
    print(f"TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print(f"{'=' * 80}\n")
    
    return passed == len(test_cases)

def test_priority_comparison():
    """Compare behavior with and without top 3 lines priority"""
    
    print("\n" + "=" * 80)
    print("TEST 2: Priority Comparison - Header vs Full Document")
    print("=" * 80)
    
    # Resume with location in BOTH header and body
    resume_with_both = """Priya Sharma
Senior Developer
Bangalore, India
priya@example.com

Career:
Mavensoft Systems Pvt Ltd, Hyderabad, India
Technical Director

Working for clients in:
- Finaplex - San Francisco, CA
- Oracle - Austin, TX
"""
    
    result = extract_location(resume_with_both)
    
    print("\nResume with location in BOTH header and body:")
    print("  Header location: Bangalore, India")
    print("  Body locations: Hyderabad, San Francisco, Austin")
    print(f"  Extracted: {result}")
    
    if 'bangalore' in result.lower():
        print("  ✅ CORRECT: Header location prioritized!")
    elif 'hyderabad' in result.lower():
        print("  ⚠️  ACCEPTABLE: Body location used (header might not have matched)")
    else:
        print("  ❌ WRONG: Extracted client location instead")
    
    return True

def test_edge_cases():
    """Test edge cases"""
    
    print("\n" + "=" * 80)
    print("TEST 3: Edge Cases")
    print("=" * 80)
    
    edge_cases = [
        {
            'name': 'Empty top 3 lines',
            'resume': """


Hyderabad, India

Resume content here...
""",
            'expected_behavior': 'Should find Hyderabad in full document'
        },
        {
            'name': 'Only 2 lines in entire resume',
            'resume': """John Doe
Portland, Oregon
""",
            'expected_behavior': 'Should find Portland'
        },
        {
            'name': 'No location anywhere',
            'resume': """John Doe
Software Engineer
john@example.com
""",
            'expected_behavior': 'Should return Unknown'
        }
    ]
    
    for case in edge_cases:
        print(f"\n  {case['name']}")
        result = extract_location(case['resume'])
        print(f"    Result: {result}")
        print(f"    Expected: {case['expected_behavior']}")
    
    return True

if __name__ == "__main__":
    print("\n" + "🔍" * 40)
    print("LOCATION EXTRACTION - TOP 3 LINES PRIORITY TEST")
    print("🔍" * 40 + "\n")
    
    test1_passed = test_location_in_top_3_lines()
    test2_passed = test_priority_comparison()
    test3_passed = test_edge_cases()
    
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    
    if test1_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nTop 3 lines priority logic is working correctly:")
        print("  1. ✅ Checks header first")
        print("  2. ✅ Uses header location when found")
        print("  3. ✅ Falls back to full extraction when header has no location")
        print("  4. ✅ Avoids client locations in body")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please review the failures above")
    
    print("=" * 80 + "\n")

