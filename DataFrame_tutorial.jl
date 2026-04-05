import Pkg
Pkg.add("DataFrames")

using DataFrames

space_company_names_array = ["SpaceX", "Blue Origin", "Boeing"]

df = DataFrame(
	space_companies = space_company_names_array
)


vehicles_array = ["car", "motorbike", "tricycle"]

wheel_count_array = [4, 2, 3]

df = DataFrame(
	vehicles = vehicles_array, 
	wheel_count = wheel_count_array
)

# Construct the dataframe out of two arrays
# (This part is similar to the previous example)

animals_array = ["dog", "spider", "chicken", "spider"]

leg_count_array = [4, 8, 2, 8]

df = DataFrame(
	animals = animals_array, 
	leg_count = leg_count_array
)

unique!(df)
deduped_df = unique(df)


inventory = DataFrame(
	item = [
		"Mars Rover",
		"Venus Explorer",
		"%Lunar Rover",
		"30% Sun Shade"
	],
	id = [
		100,
		101,
		102,
		103
	],
	kind = [
		"rover",
		"spaceship",
		"rover",
		"Sun Shade"
	]
)

for i in eachrow(inventory)
	i[:kind] = replace(i[:kind], "rover"=>"Rover")
	i[:kind] = replace(i[:kind], "spaceship"=>"Spaceship")
end


for i in eachrow(inventory)
	i[:item] = replace(i[:item], r"^%"=>"")
end

rovers = filter(
	x -> any(occursin.(["Rover"], x.item)),
	inventory
)