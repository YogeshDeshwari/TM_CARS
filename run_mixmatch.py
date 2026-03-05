from style_compiler import StyleCompiler, make_mixmatch_recipe


def main():
    compiler = StyleCompiler(size=2048)

    # Generate a small batch of novel mixed liveries
    for i in range(1, 9):
        seed = 1000 + i * 77
        recipe = make_mixmatch_recipe(
            name=f"MixMatch_{i:02d}",
            seed=seed,
            motif_stack=None,  # default fade/topo/halftone/glitch
        )
        engine = compiler.render(recipe, team_name=recipe.name)
        engine.set_dirt_amount(recipe.dirt_amount)
        engine.save()


if __name__ == "__main__":
    main()

