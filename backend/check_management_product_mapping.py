"""Check which roles map to Management & Product"""

from role_processor import RoleProcessor

config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Reset@123',
    'database': 'ats_db',
    'port': 3306
}

with RoleProcessor(config=config) as rp:
    roles = rp.get_all_original_roles('Management & Product')
    print("=" * 70)
    print("Roles that map to 'Management & Product':")
    print("=" * 70)
    print(f"Total: {len(roles)} roles\n")
    for i, role in enumerate(roles, 1):
        print(f"{i:2}. {role}")
    
    print("\n" + "=" * 70)
    print("Testing some role matches:")
    print("=" * 70)
    
    test_roles = [
        "Project Manager",
        "Program Manager",
        "Technical Director",
        "VP Engineering",
        "CEO",
        "CTO",
        "Delivery Manager",
        "Senior Manager"
    ]
    
    for role in test_roles:
        normalized = rp.normalize_role_from_resume(role)
        print(f"'{role}' -> '{normalized}'")

