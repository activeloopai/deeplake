from hub.core.storage.gdrive import GDriveProvider

gdrive = GDriveProvider("1eciJqVgSwA69APhUZO0o1UJBUbF6ZBon")
gdrive["c/a/v"] = (1020).to_bytes(3, "little")
print(int.from_bytes(gdrive["c/a/v"], "little"))
gdrive["c/a/g"] = (2048).to_bytes(4, "little")
print(int.from_bytes(gdrive["c/a/g"], "little"))
print(gdrive._all_keys())
