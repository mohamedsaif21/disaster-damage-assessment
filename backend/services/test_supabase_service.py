from backend.services.supabase_service import get_supabase


def main():
    supabase = get_supabase()

    response = (
        supabase
        .table("assessment_models")
        .select("*")
        .limit(5)
        .execute()
    )

    print("Supabase connection successful.")
    print("Rows returned:", len(response.data))
    print("Data:", response.data)


if __name__ == "__main__":
    main()